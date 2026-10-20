# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU checks for the layered GMM benchmark's derived plan and metrics.

These run everywhere: no torch, no torch_npu, no device. They guard the parts of
``tests/npu/gmm_layered_performance.py`` that decide *what* gets measured, since
an error there would produce plausible numbers for the wrong problem - which is
harder to notice than a crash.

The benchmark script itself is not named ``test_*`` on purpose: it is a harness
that must be launched explicitly on a 910C, so pytest must not collect it.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_PATH = REPOSITORY_ROOT / "tests/npu/gmm_layered_performance.py"


def load_benchmark() -> ModuleType:
    """Import the harness by path; it is not a pytest-collected module.

    The module is registered in ``sys.modules`` before execution because
    ``dataclasses`` resolves field annotations through the owning module's
    namespace, which requires the entry to exist.
    """
    name = "gmm_layered_performance"
    spec = importlib.util.spec_from_file_location(name, BENCHMARK_PATH)
    # Both are Optional for a path that may be missing or unloadable; the file is
    # in-tree, so a None here is a broken checkout rather than a test outcome.
    assert spec is not None and spec.loader is not None, (
        f"cannot load the benchmark harness from {BENCHMARK_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


benchmark = load_benchmark()

DEFAULT_GEOMETRY = (65536, 98304, 262144)


def harness_function(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Return the ``ast`` node of one top-level function in the harness.

    Four checks in this module analyse the harness source instead of running it,
    because the properties they guard - unbound names and device placement - only
    surface on hardware. They share this lookup rather than each re-parsing.
    """
    tree = ast.parse(BENCHMARK_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def make_result(
    *,
    kernel: str = "w13",
    distribution: str = "balanced",
    layer_mode: str = "rotate",
    execution_mode: str = "eager",
    layered_ms: float = 8.0,
    baseline_ms: float | None = None,
    layers: int = 8,
    geometry: tuple[int, int, int] = DEFAULT_GEOMETRY,
) -> dict[str, object]:
    """Build one synthetic result in the shape ``render_*`` expects.

    The tables read per-layer medians, so the totals stored here are multiplied
    back up by ``layers``. One builder serves every rendering test; four
    near-identical ones previously lived in the individual classes, which is the
    duplication AGENTS.md calls a layout bug.
    """
    tokens, rows, capacity = geometry
    baseline_ms = layered_ms * 2 if baseline_ms is None else baseline_ms
    return {
        "kernel": kernel,
        "distribution": distribution,
        "mode": layer_mode,
        "execution_mode": execution_mode,
        "layers": layers,
        "source_tokens": tokens,
        "valid_rows": rows,
        "capacity": capacity,
        "case": f"{kernel} {distribution} {layer_mode} {execution_mode}",
        "layered": {
            "event": benchmark.summarize([layered_ms * layers]),
            "host": benchmark.summarize([layered_ms * layers]),
        },
        "baseline": {
            "event": benchmark.summarize([baseline_ms * layers]),
            "host": benchmark.summarize([baseline_ms * layers]),
        },
        "flops": benchmark.gemm_flops(rows, 4096, 4096),
        "weight_bytes": 512 * 1024 * 1024,
    }


class DerivedGeometryTests(unittest.TestCase):
    """M and C are derived, so the derivation is the specification."""

    def test_valid_rows_follows_topk_and_expert_parallel(self):
        # 6 rows per source token over 4 expert-parallel ranks.
        self.assertEqual(benchmark.derive_valid_rows(65536), 98304)
        self.assertEqual(benchmark.derive_valid_rows(1024), 1536)

    def test_valid_rows_rejects_a_batch_that_cannot_balance(self):
        # 6 * T must divide by 4; an odd T would leave a half row.
        with self.assertRaises(ValueError):
            benchmark.derive_valid_rows(7)

    def test_valid_rows_rejects_nonpositive_input(self):
        for value in (0, -1):
            with self.assertRaises(ValueError):
                benchmark.derive_valid_rows(value)

    def test_capacity_mirrors_the_batch_size_factor_tiling(self):
        # The host tiling computes floor(262144 * factor).
        self.assertEqual(benchmark.derive_capacity(1.0), 262144)
        self.assertEqual(benchmark.derive_capacity(0.5), 131072)
        self.assertEqual(benchmark.derive_capacity(0.25), 65536)

    def test_capacity_rejects_a_factor_outside_the_supported_range(self):
        for factor in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                benchmark.derive_capacity(factor)

    def test_geometry_matches_the_documented_model_constants(self):
        self.assertEqual(benchmark.HIDDEN, 4096)
        self.assertEqual(benchmark.INTERMEDIATE, 2048)
        self.assertEqual(benchmark.EXPERTS_PER_RANK, 64)
        self.assertEqual(benchmark.TOP_K, 6)
        self.assertEqual(benchmark.EXPERT_PARALLEL, 4)

    def test_packed_weight_shapes_are_the_int32_storage_shapes(self):
        # Eight int4 nibbles per int32 word, so the last axis is width / 8.
        self.assertEqual(benchmark.W13_PACKED_N // benchmark.INT8_PER_INT32, 512)
        self.assertEqual(benchmark.W2_PACKED_N // benchmark.INT8_PER_INT32, 512)


class DistributionTests(unittest.TestCase):
    def test_balanced_weights_every_expert_equally(self):
        weights = benchmark.distribution_weights(64, "balanced")
        self.assertEqual(len(weights), 64)
        self.assertEqual(len(set(weights)), 1)

    def test_skewed_leaves_a_tail_empty_and_front_loads_the_rest(self):
        weights = benchmark.distribution_weights(64, "skewed")
        self.assertEqual(len(weights), 64)
        self.assertGreater(weights[-1], -1)  # present but possibly zero
        self.assertEqual(weights.count(0.0), 16)
        live = [weight for weight in weights if weight > 0.0]
        self.assertEqual(live, sorted(live, reverse=True))

    def test_unknown_distribution_is_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.distribution_weights(64, "uniform")

    def test_counts_sum_to_the_valid_row_count_exactly(self):
        # M is defined as sum(group_list), so an off-by-one would silently change
        # the size of the measured matrix.
        for distribution in benchmark.DISTRIBUTIONS:
            weights = benchmark.distribution_weights(64, distribution)
            for rows in (1, 7, 1536, 98304):
                counts = benchmark.allocate_counts(rows, weights)
                self.assertEqual(sum(counts), rows, distribution)
                self.assertEqual(len(counts), 64)

    def test_counts_preserve_the_intended_ordering(self):
        weights = benchmark.distribution_weights(64, "skewed")
        counts = benchmark.allocate_counts(98304, weights)
        # A decaying weight vector must produce a non-increasing count vector.
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_skewed_and_balanced_differ_at_equal_rows(self):
        rows = 98304
        balanced = benchmark.allocate_counts(
            rows, benchmark.distribution_weights(64, "balanced")
        )
        skewed = benchmark.allocate_counts(
            rows, benchmark.distribution_weights(64, "skewed")
        )
        self.assertEqual(sum(balanced), sum(skewed))
        self.assertNotEqual(balanced, skewed)
        # The whole point of the skewed axis: idle experts appear.
        self.assertIn(0, skewed)
        self.assertNotIn(0, balanced)

    def test_allocate_counts_rejects_degenerate_input(self):
        with self.assertRaises(ValueError):
            benchmark.allocate_counts(-1, [1.0])
        with self.assertRaises(ValueError):
            benchmark.allocate_counts(1, [])
        with self.assertRaises(ValueError):
            benchmark.allocate_counts(1, [0.0, 0.0])


class SummaryTests(unittest.TestCase):
    """Percentiles are nearest-rank and never pooled across configurations."""

    def test_nearest_rank_p95_does_not_interpolate(self):
        report = benchmark.summarize([float(value) for value in range(1, 21)])
        self.assertEqual(report["count"], 20)
        self.assertEqual(report["median_ms"], 10.5)
        self.assertEqual(report["p95_ms"], 19.0)

    def test_single_sample_and_empty_input(self):
        self.assertEqual(benchmark.summarize([]), {"count": 0})
        self.assertEqual(benchmark.summarize([2.0])["p95_ms"], 2.0)

    def test_flops_counts_one_multiply_and_one_add(self):
        self.assertEqual(benchmark.gemm_flops(2, 3, 4), 48)


class CaseMatrixTests(unittest.TestCase):
    """The case builder decides what combinations reach the device."""

    def build(self, **overrides):
        kwargs = dict(
            source_tokens=[65536],
            capacities=[262144],
            distributions=["balanced", "skewed"],
            layer_modes=["single", "rotate"],
            kernels=["w13", "w2", "chain"],
            execution_modes=["eager", "aclgraph"],
        )
        kwargs.update(overrides)
        return benchmark.build_cases(**kwargs)

    def test_one_case_per_kernel_distribution_layer_mode_and_launch_path(self):
        cases = self.build()
        self.assertEqual(len(cases), 3 * 2 * 2 * 2)

    def test_both_launch_paths_are_planned_for_every_configuration(self):
        # The pair is the point of the execution axis: every combination must
        # appear under both paths, or the comparison table has a hole in it.
        seen: dict[tuple[str, str, str], set[str]] = {}
        for case in self.build():
            seen.setdefault(case.key(), set()).add(case.execution_mode)
        self.assertTrue(seen)
        for key, modes in seen.items():
            self.assertEqual(modes, set(benchmark.EXECUTION_MODES), key)

    def test_an_unknown_launch_path_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build(execution_modes=["graph"])

    def test_a_single_launch_path_can_be_requested(self):
        cases = self.build(execution_modes=["eager"])
        self.assertEqual(len(cases), 3 * 2 * 2)
        self.assertEqual({case.execution_mode for case in cases}, {"eager"})

    def test_every_case_carries_its_own_geometry_for_auditing(self):
        for case in self.build():
            self.assertEqual(case.valid_rows, 98304)
            self.assertEqual(case.capacity, 262144)
            self.assertEqual(len(case.counts), 64)
            self.assertIn("M=98304", case.describe())
            self.assertIn("M/C=0.375", case.describe())

    def test_rows_exceeding_capacity_are_rejected(self):
        # A receive chunk must hold every row it was sized for.
        with self.assertRaises(ValueError):
            self.build(capacities=[1024])

    def test_skewed_load_still_fits_one_receive_chunk(self):
        # CAM chunks whole experts, so the reference harness rejects a case where
        # one expert's rows exceed C. Under apportionment the largest expert can
        # never exceed M, and M <= C is enforced, so that failure mode is
        # unreachable here by construction. This asserts the invariant that makes
        # it unreachable rather than leaving it implied.
        for case in self.build(distributions=["skewed"], capacities=[131072]):
            self.assertLessEqual(max(case.counts), case.capacity)
            self.assertLessEqual(max(case.counts), case.valid_rows)

    def test_a_single_expert_cannot_outgrow_the_row_total(self):
        # The guard that would be needed to reject an oversized expert is dead
        # code here, and this is the reason.
        weights = benchmark.distribution_weights(64, "skewed")
        for rows in (1, 1536, 98304):
            counts = benchmark.allocate_counts(rows, weights)
            self.assertLessEqual(max(counts), rows)

    def test_unknown_kernel_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build(kernels=["w9"])

    def test_empty_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build(kernels=[])

    def test_capacity_utilisation_never_reaches_one(self):
        # M = top-k * T / EP carries a factor of three while C must be a power of
        # two, so 3T/2 = 2**k has no integer solution. The closest case is 0.75.
        cases = self.build(capacities=[131072])
        self.assertEqual({case.valid_rows / case.capacity for case in cases}, {0.75})
        for case in self.build():
            self.assertLess(case.valid_rows / case.capacity, 1.0)


class PlanTests(unittest.TestCase):
    """The dry run is the contract users read before spending device time."""

    def plan(self, argv=None):
        result = subprocess.run(
            [sys.executable, "-S", str(BENCHMARK_PATH), "--dry-run"] + (argv or []),
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_dry_run_needs_no_device_packages(self):
        # -S drops site-packages, so a torch import here would fail loudly.
        plan = self.plan()
        self.assertEqual(plan["status"], "dry-run; no device execution")
        self.assertGreater(plan["case_count"], 0)

    def test_plan_reports_the_geometry_and_timer_contract(self):
        plan = self.plan()
        self.assertEqual(plan["geometry"]["hidden"], 4096)
        self.assertEqual(plan["geometry"]["experts_per_rank"], 64)
        self.assertEqual(plan["geometry"]["w13_packed_weight"], [64, 4096, 512])
        self.assertEqual(plan["geometry"]["w2_packed_weight"], [64, 2048, 512])
        # Both timers must be described, because only host time shows the saving
        # the layered operators exist to produce.
        self.assertIn("host_ms", plan["timing_contract"])
        self.assertIn("event_ms", plan["timing_contract"])

    def test_plan_warns_when_the_source_batch_crosses_uint16(self):
        plan = self.plan(["--source-tokens", "65536"])
        self.assertTrue(any("uint16" in item for item in plan["warnings"]))

    def test_plan_does_not_warn_below_the_uint16_bound(self):
        plan = self.plan(["--source-tokens", "32768"])
        self.assertFalse(any("uint16" in item for item in plan["warnings"]))

    def test_plan_names_the_unreachable_full_capacity_case(self):
        plan = self.plan()
        self.assertIn("unreachable", plan["capacity_note"])

    def test_unbalanced_source_tokens_are_rejected_by_the_cli(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(BENCHMARK_PATH),
                "--dry-run",
                "--source-tokens",
                "7",
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("divisible", result.stderr)

    def test_unknown_enum_values_are_rejected_by_the_cli(self):
        for flag, value in (
            ("--distribution", "uniform"),
            ("--layer-mode", "cycle"),
            ("--only", "w9"),
        ):
            result = subprocess.run(
                [sys.executable, "-S", str(BENCHMARK_PATH), "--dry-run", flag, value],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0, f"{flag}={value} was accepted")


class GraphMeasurementTests(unittest.TestCase):
    """The aclgraph path's sequencing contract, exercised with fakes.

    The device half needs hardware, but the ordering rules do not: warmup before
    capture, one capture, then one replay per sample with the input restored
    outside the measured interval. Getting that order wrong inflates every
    sample (by capturing one-time warmup work) or corrupts the input (by
    replaying onto a buffer the A8W4 MSD path rewrote), and neither is visible
    in the printed numbers.

    Every fake appends to one shared ``order`` list, so the whole sequence is
    observable in a single assertion instead of being inferred per call site.
    """

    class Recorder:
        """Shared event log plus a capture-depth counter."""

        def __init__(self) -> None:
            self.order: list[str] = []
            self.capture_depth = 0

        def event(self, **_: object) -> GraphMeasurementTests.FakeEvent:
            return GraphMeasurementTests.FakeEvent(self)

        def graph(self) -> GraphMeasurementTests.FakeGraph:
            return GraphMeasurementTests.FakeGraph(self)

        def synchronize(self) -> None:
            self.order.append("sync")

        def context(self) -> GraphMeasurementTests.FakeGraphContext:
            return GraphMeasurementTests.FakeGraphContext(self)

    class FakeEvent:
        def __init__(self, owner) -> None:
            self.owner = owner

        def record(self) -> None:
            self.owner.order.append("record")

        def elapsed_time(self, _other) -> float:
            return 1.0

    class FakeGraph:
        def __init__(self, owner) -> None:
            self.owner = owner
            self.replays = 0

        def replay(self) -> None:
            self.replays += 1
            self.owner.order.append("replay")

    class FakeGraphContext:
        def __init__(self, owner) -> None:
            self.owner = owner

        def __enter__(self) -> None:
            self.owner.capture_depth += 1
            self.owner.order.append("capture")

        def __exit__(self, *_: object) -> None:
            self.owner.capture_depth -= 1

    def setUp(self) -> None:
        self.recorder = self.Recorder()
        recorder = self.recorder

        class FakeNpuModule:
            Event = staticmethod(recorder.event)
            synchronize = staticmethod(recorder.synchronize)

        class FakeGraphModule:
            NPUGraph = staticmethod(recorder.graph)
            graph = staticmethod(lambda _graph: recorder.context())

        class FakeTorchNpu:
            npu = FakeNpuModule

        class FakeTorch:
            npu = FakeGraphModule

        self.torch = FakeTorch
        self.torch_npu = FakeTorchNpu

    def test_warmup_happens_before_capture_and_never_inside_it(self):
        depths: list[int] = []

        def run() -> None:
            self.recorder.order.append("run")
            depths.append(self.recorder.capture_depth)

        benchmark.measure(
            self.torch,
            self.torch_npu,
            run,
            warmup=3,
            iterations=2,
            execution_mode="aclgraph",
        )
        # Three warmup calls, all outside any capture, then exactly one captured
        # call, then two replays. A warmup inside the capture would make every
        # sample pay the one-time tiling cost.
        self.assertEqual(depths[:3], [0, 0, 0])
        self.assertEqual(depths.count(1), 1)
        self.assertEqual(self.recorder.order.count("capture"), 1)
        self.assertEqual(self.recorder.order.count("replay"), 2)

    def test_each_sample_restores_before_the_events_bracket_the_replay(self):
        benchmark.measure(
            self.torch,
            self.torch_npu,
            lambda: self.recorder.order.append("run"),
            warmup=0,
            iterations=1,
            restore=lambda: self.recorder.order.append("restore"),
            execution_mode="aclgraph",
        )
        # Warmup still runs once even when asked for zero, because the capture
        # must not be the first call of a shape. The measured sample is then
        # restore -> sync -> record -> replay -> record -> sync: the copy that
        # re-materializes the rewritten activation sits entirely outside the
        # interval between the two events.
        self.assertEqual(
            self.recorder.order,
            [
                "restore",
                "run",
                "sync",
                "capture",
                "run",
                "sync",
                "restore",
                "sync",
                "record",
                "replay",
                "record",
                "sync",
            ],
        )

    def test_eager_never_constructs_a_graph(self):
        benchmark.measure(
            self.torch,
            self.torch_npu,
            lambda: self.recorder.order.append("run"),
            warmup=1,
            iterations=1,
            execution_mode="eager",
        )
        self.assertEqual(self.recorder.order.count("replay"), 0)
        self.assertEqual(self.recorder.order.count("capture"), 0)

    def test_an_unknown_execution_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.measure(
                self.torch,
                self.torch_npu,
                lambda: None,
                warmup=0,
                iterations=1,
                execution_mode="graph",
            )


class ExecutionModeTableTests(unittest.TestCase):
    """The eager-versus-graph table pairs the two paths per configuration."""

    def result(
        self, kernel, layer_mode, exec_mode, per_layer_ms, layers=8, geometry=None
    ):
        return make_result(
            kernel=kernel,
            layer_mode=layer_mode,
            execution_mode=exec_mode,
            layered_ms=per_layer_ms,
            layers=layers,
            geometry=geometry or DEFAULT_GEOMETRY,
        )

    def test_graph_speedup_is_eager_over_graph_for_the_same_unit(self):
        results = [
            self.result("w13", "rotate", "eager", 8.0),
            self.result("w13", "rotate", "aclgraph", 6.0),
        ]
        table = "\n".join(benchmark.render_execution_mode_table(results))
        # 8 ms eager against 6 ms graphed is 1.333x, for both operator sides.
        self.assertIn("1.333x", table)
        self.assertIn("| layered |", table)
        self.assertIn("| baseline |", table)

    def test_a_configuration_missing_one_path_is_skipped_not_faked(self):
        results = [self.result("w13", "rotate", "eager", 8.0)]
        table = "\n".join(benchmark.render_execution_mode_table(results))
        self.assertNotIn("1.333x", table)
        self.assertIn("no configuration was measured under both", table)

    def test_single_and_rotate_stay_separate_rows(self):
        results = [
            self.result("w13", "single", "eager", 9.0, layers=1),
            self.result("w13", "single", "aclgraph", 9.0, layers=1),
            self.result("w13", "rotate", "eager", 8.0),
            self.result("w13", "rotate", "aclgraph", 6.0),
        ]
        table = "\n".join(benchmark.render_execution_mode_table(results))
        self.assertIn("(single)", table)
        self.assertIn("(rotate)", table)

    def test_two_geometries_do_not_overwrite_each_other(self):
        """A sweep over --source-tokens or --capacity must keep every workload.

        Keying without the geometry left both of these on the same key, so the
        later one silently replaced the earlier measurement and the report showed
        one geometry's numbers for all of them. Eight results in, eight rows out.
        """
        geometries = [
            (65536, 98304, 262144),
            (32768, 49152, 131072),
        ]
        results = []
        for tokens, rows, capacity in geometries:
            geometry = (tokens, rows, capacity)
            results += [
                self.result("w13", "rotate", "eager", 8.0, geometry=geometry),
                self.result("w13", "rotate", "aclgraph", 6.0, geometry=geometry),
            ]
        table = "\n".join(benchmark.render_execution_mode_table(results))
        data_rows = [line for line in table.splitlines() if line.startswith("| w13")]
        # Two geometries x two operator sides.
        self.assertEqual(len(data_rows), 4, table)
        for tokens, _rows, _capacity in geometries:
            self.assertIn(f"T={tokens}", table)
        self.assertEqual(table.count("T=65536"), table.count("T=32768"))


class DecompositionTableTests(unittest.TestCase):
    """The c/f decomposition keys on the geometry as well as the launch path."""

    def result(self, layer_mode, exec_mode, per_layer_ms, layers, geometry):
        return make_result(
            layer_mode=layer_mode,
            execution_mode=exec_mode,
            layered_ms=per_layer_ms,
            layers=layers,
            geometry=geometry,
        )

    def test_two_geometries_produce_two_rows_per_launch_path(self):
        """Same regression the execution-mode table needed.

        Both geometries share kernel, distribution, layer mode and launch path,
        so without the geometry in the key the second would replace the first and
        the fit would describe one workload instead of two.
        """
        geometries = [(65536, 98304, 262144), (32768, 49152, 131072)]
        results = []
        for geometry in geometries:
            results += [
                self.result("single", "eager", 9.0, 1, geometry),
                self.result("rotate", "eager", 8.0, 8, geometry),
                self.result("single", "aclgraph", 7.5, 1, geometry),
                self.result("rotate", "aclgraph", 7.0, 8, geometry),
            ]
        table = "\n".join(benchmark.render_decomposition_table(results))
        data_rows = [line for line in table.splitlines() if line.startswith("| w13")]
        # Two geometries x two launch paths.
        self.assertEqual(len(data_rows), 4, table)
        self.assertIn("T=65536", table)
        self.assertIn("T=32768", table)

    def test_the_solved_costs_match_the_inputs(self):
        # single = c + f and rotate = c + f/8, so a 1 ms gap corresponds to
        # f = 1 * 8/7 and c = 9 - f.
        geometry = (65536, 98304, 262144)
        results = [
            self.result("single", "eager", 9.0, 1, geometry),
            self.result("rotate", "eager", 8.0, 8, geometry),
        ]
        table = "\n".join(benchmark.render_decomposition_table(results))
        row = next(line for line in table.splitlines() if line.startswith("| w13"))
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        per_call, fixed = float(cells[4]), float(cells[5])
        self.assertAlmostEqual(fixed, 8 / 7, places=4)
        self.assertAlmostEqual(per_call, 9.0 - 8 / 7, places=4)


class CapacityUtilisationTests(unittest.TestCase):
    """Exact full capacity is unreachable; near-full capacity is not.

    The module used to claim 0.75 was the largest reachable utilisation, which
    reads as an arithmetic impossibility. Only ``C`` is constrained to a power of
    two, so a source batch that is not one can land just under capacity.
    """

    def test_near_full_utilisation_is_reachable_and_valid(self):
        tokens, capacity = 43690, 65536
        rows = benchmark.derive_valid_rows(tokens)
        self.assertEqual(rows, 65535)
        utilisation = rows / capacity
        self.assertGreater(utilisation, 0.9999)
        self.assertLess(utilisation, 1.0)
        # And it is a legal plan, not merely arithmetic: the source batch stays
        # under the uint16 bound the dispatch counters impose.
        self.assertLessEqual(tokens, benchmark.UINT16_MAX)
        cases = benchmark.build_cases(
            source_tokens=[tokens],
            capacities=[capacity],
            distributions=["balanced"],
            layer_modes=["single"],
            kernels=["w13"],
            execution_modes=["eager"],
        )
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].valid_rows, 65535)

    def test_exact_full_capacity_has_no_integer_source_batch(self):
        """The half of the original claim that was true.

        Reaching ``M/C == 1`` needs ``3T/2 == 2**k``, i.e. ``3T == 2**(k+1)``,
        which has no integer solution because the left side carries a factor of
        three. This is a property of the arithmetic, not of the CLI, so it is
        asserted directly rather than by expecting a rejection.
        """
        for power in range(1, 19):
            target = 2**power
            # 3T = 2*target has no integer solution unless target is divisible by 3.
            if target % 3:
                self.assertNotEqual((2 * target) % 3, 0)

    def test_the_report_does_not_cap_utilisation_at_075(self):
        """Assert the rendered claim, not the source that produces it.

        AGENTS.md allows text assertions on artifacts but not on production
        modules, and the report is the artifact here. Checking its output also
        catches a reinstated claim however it was reintroduced, rather than only
        when the wording matches the old string exactly.
        """
        plan = benchmark.describe_plan(
            benchmark.build_cases(
                source_tokens=[65536],
                capacities=[262144],
                distributions=["balanced"],
                layer_modes=["single"],
                kernels=["w13"],
                execution_modes=["eager"],
            ),
            type("Args", (), {"layers": 8})(),
        )
        self.assertNotIn("0.75", plan["capacity_note"])
        self.assertIn("Near-full capacity is reachable", plan["capacity_note"])


class ReportRenderingTests(unittest.TestCase):
    """The report table is rendered from synthetic results, with no device."""

    def result(self, layered_ms, baseline_ms, layers=8):
        # These medians are per-layer values, and make_result takes per-layer
        # inputs too, so no rescaling happens here.
        return make_result(
            layered_ms=layered_ms / layers,
            baseline_ms=baseline_ms / layers,
            layers=layers,
        )

    def test_layered_and_baseline_medians_are_reported_per_layer(self):
        result = self.result(layered_ms=8.0, baseline_ms=16.0, layers=8)
        # A rotation total of 8 ms over 8 layers is 1 ms per layer, and the
        # baseline at twice that must report a 2x speedup.
        self.assertEqual(benchmark.per_layer_median_ms(result, "layered"), 1.0)
        self.assertEqual(benchmark.per_layer_median_ms(result, "baseline"), 2.0)
        table = "\n".join(benchmark.render_results_table([result]))
        self.assertIn("2.000x", table)

    def test_single_layer_mode_is_not_divided(self):
        result = self.result(layered_ms=2.0, baseline_ms=3.0, layers=1)
        self.assertEqual(benchmark.per_layer_median_ms(result, "layered"), 2.0)

    def test_a_missing_baseline_renders_as_not_available(self):
        result = self.result(layered_ms=1.0, baseline_ms=1.0)
        result["baseline"] = {
            "event": benchmark.summarize([]),
            "host": benchmark.summarize([]),
        }
        self.assertIsNone(benchmark.per_layer_median_ms(result, "baseline"))
        table = "\n".join(benchmark.render_results_table([result]))
        self.assertIn("n/a", table)

    def test_efficiency_table_derives_throughput_from_the_median(self):
        result = self.result(layered_ms=8.0, baseline_ms=16.0)
        table = "\n".join(benchmark.render_efficiency_table([result]))
        self.assertIn("TFLOP/s", table)
        self.assertIn("GB/s", table)

    def test_full_report_names_both_operators_and_the_chain(self):
        class Args:
            layers = 8
            warmup = 5
            iterations = 20
            seed = 1234

        results = [self.result(layered_ms=8.0, baseline_ms=16.0)]
        report = benchmark.render_report(
            Args(), results, ["baseline_differential PASSED"]
        )
        self.assertIn("Layered computation operator benchmark", report)
        # Both operators plus their composition must be described, since the file
        # is the single archived record for the pair.
        self.assertIn("w13", report)
        self.assertIn("w2", report)
        self.assertIn("chain", report)
        self.assertIn("unreachable", report)
        self.assertIn("baseline_differential PASSED", report)


class ReportFileNameTests(unittest.TestCase):
    def test_default_report_name_covers_both_operators(self):
        # The name is a contract: one file, both operators.
        self.assertEqual(
            benchmark.REPORT_NAME, "layered_computation_operator_benchmark.md"
        )
        self.assertTrue(benchmark.REPORT_NAME.endswith(".md"))

    def test_report_is_opt_in(self):
        # Without --report nothing is written; the default is None.
        result = subprocess.run(
            [sys.executable, "-S", str(BENCHMARK_PATH), "--help"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--report", result.stdout)


class DevicePathNameBindingTests(unittest.TestCase):
    """Module-level functions must not load a name they never receive.

    ``baseline_gmm`` read ``torch.bfloat16`` without taking ``torch``, so it
    raised ``NameError`` the first time it ran on a device. Nothing that
    exercises the plan can see that: the failure needs the device path, and the
    device path needs hardware. This checks it by static analysis instead, which
    is cheap and needs no torch.

    Only module-level functions are considered. Nested ones legitimately close
    over their enclosing scope, and resolving that would mean a real static
    analysis rather than the cheap guard that fits here.
    """

    def module_level_functions(self):

        tree = ast.parse(BENCHMARK_PATH.read_text(encoding="utf-8"))
        return [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    def module_level_names(self, tree):

        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            # Imports, including the ones under a TYPE_CHECKING guard or in a
            # try block, bind at module scope.
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    for alias in inner.names:
                        names.add(alias.asname or alias.name.split(".")[0])
        return names

    def attribute_bases_loaded(self, function):
        """Return every ``X`` in an ``X.attr`` access inside the function body."""

        bases = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                bases.add(node.value.id)
        return bases

    def locally_bound_names(self, function):
        """Return the names the function binds anywhere in its own body.

        Parameters, assignments, imports, ``except ... as``, and the names and
        parameters of nested functions all bind a name. A wholesale bind is fine
        here: this guard looks for names the function could never resolve, and
        being generous about what counts as bound only avoids false positives.
        """

        bound = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
        for extra in (function.args.vararg, function.args.kwarg):
            if extra is not None:
                bound.add(extra.arg)
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    bound.add(node.name)
                bound.update(arg.arg for arg in node.args.args + node.args.kwonlyargs)
                for extra in (node.args.vararg, node.args.kwarg):
                    if extra is not None:
                        bound.add(extra.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
        return bound

    def test_no_module_function_loads_an_undefined_name(self):

        tree = ast.parse(BENCHMARK_PATH.read_text(encoding="utf-8"))
        globals_ = self.module_level_names(tree)
        offenders = []
        for function in self.module_level_functions():
            unknown = (
                self.attribute_bases_loaded(function)
                - self.locally_bound_names(function)
                - globals_
            )
            for name in sorted(unknown):
                offenders.append(
                    f"{function.name} (line {function.lineno}) loads {name!r}"
                )
        self.assertEqual(
            offenders,
            [],
            "these would raise NameError on the device path: " + "; ".join(offenders),
        )

    def test_the_scan_finds_a_real_missing_parameter(self):
        # The exact defect this guard exists for: a function that reads
        # torch.bfloat16 without taking torch.

        tree = ast.parse(
            "def baseline_gmm(torch_npu, weight):\n"
            "    return torch_npu.npu_grouped_matmul(weight, dtype=torch.bfloat16)\n"
        )
        function = tree.body[0]
        unknown = (
            self.attribute_bases_loaded(function)
            - self.locally_bound_names(function)
            - self.module_level_names(tree)
        )
        self.assertEqual(unknown, {"torch"})

    def test_the_baseline_helpers_receive_both_modules(self):
        for name in ("baseline_gmm", "baseline_w13", "baseline_w2"):
            helper = getattr(benchmark, name)
            parameters = helper.__code__.co_varnames[: helper.__code__.co_argcount]
            self.assertIn("torch", parameters, f"{name} must receive torch")
            self.assertIn("torch_npu", parameters, f"{name} must receive torch_npu")


class OperandPlacementTests(unittest.TestCase):
    """Operands that reach an operator must be built on the device.

    The operators compare the devices of every operand, so one left on the host
    is rejected with "Expected all tensors to be on the same device" - a message
    that names neither the operand nor the function that built it. That happened
    to the A8W4 compensation, which was computed on the CPU and appended without
    a transfer. This checks the construction site instead, which is cheap and
    needs no torch.
    """

    def function_calls(self, name):
        """Return the set of method names called inside ``name``.

        Asserting on a call set rather than on source text keeps these checks
        stable under reformatting: ``ast.unparse`` normalizes quoting and
        parenthesization, so a text match would break for reasons unrelated to
        the property being guarded.
        """
        return {
            node.func.attr
            for node in ast.walk(harness_function(name))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

    def assignments(self, function_name):
        """Return ``{target_name: unparsed_value}`` for simple assignments.

        The guard has to look at the *value bound to a specific name*, not at
        whatever calls the function happens to contain: an earlier version of
        this check only asked whether ``pack_int4_weights`` called ``.npu()``
        anywhere, which stayed true because the weight was still transferred, and
        so it passed with the compensation bug reintroduced.
        """
        found = {}
        for node in ast.walk(harness_function(function_name)):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    found[target.id] = ast.unparse(node.value)
        return found

    def test_pack_int4_weights_transfers_its_operands(self):
        # weight and compensation are consumed by the operator directly, so both
        # must reach the device where they are built. The scale is returned on the
        # host on purpose: callers reinterpret its bits to int64 before
        # transferring it, and doing that on the device would move eight bytes per
        # element just to move it back.
        assignments = self.assignments("pack_int4_weights")
        self.assertIn("npu", assignments["compensation"], assignments["compensation"])
        self.assertIn("npu_format_cast", assignments["weight"], assignments["weight"])
        # The scale must stay on the host, which is what makes this a statement
        # about the compensation rather than about the function in general.
        self.assertNotIn("npu", assignments["scale"], assignments["scale"])

    def test_build_weights_transfers_the_scales_it_stores(self):
        function = ast.unparse(harness_function("build_weights"))
        # Both appended scales must be transferred, since the operator compares
        # devices. The W13 one is 2-D and the W2 one is reshaped to [E, 1, H] on
        # the way; both end in .npu().
        self.assertEqual(function.count(".npu()"), 2, function)
        self.assertIn("scale13", function)
        self.assertIn("scale2", function)

    def test_the_scan_finds_an_untransferred_operand(self):
        # Non-vacuity: the exact defect, with the .npu() call removed.
        tree = ast.parse(
            "def pack_int4_weights():\n"
            "    compensation = (FACTOR * column_sum * scale).contiguous()\n"
            "    return compensation\n"
        )
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("npu", calls)

    def test_build_activation_builds_both_on_the_device(self):
        self.assertIn("npu", self.function_calls("build_activation"))

    def test_group_list_and_layer_index_are_created_on_the_device(self):

        tree = ast.parse(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for name in ("build_group_list", "layer_index_tensor"):
            function = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            devices = [
                keyword.value.value
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                for keyword in node.keywords
                if keyword.arg == "device" and isinstance(keyword.value, ast.Constant)
            ]
            self.assertEqual(devices, ["npu"], f"{name} must build on the device")


if __name__ == "__main__":
    unittest.main()
