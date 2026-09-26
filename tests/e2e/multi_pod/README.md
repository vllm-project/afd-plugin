# `tests/e2e/multi_pod/runner.py` walkthrough

This document explains how `runner.py` executes, using
`tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite_multi_pod.py` as the
worked example. It is a companion to the code, not a replacement for it —
line numbers refer to `runner.py` as of this writing and may drift.

## The core idea

A "multi-pod" AFD E2E scenario splits Attention and FFN ranks across more
than one pod (or container). Something has to decide, per pod, which ranks
it should launch, how those ranks should be wired to their peers in other
pods, and when it is safe to evaluate. `runner.py` is that "something" — but
deliberately with **no master process**. The same program, with the same
argv, runs once inside *every* participating pod. Each copy:

1. figures out its own identity from its environment (pod 0? pod 1? …),
2. derives the *entire* global plan locally, with a pure function, from
   inputs every pod was given identically,
3. reads out only the slice of that plan that belongs to it,
4. coordinates with its peers through a shared rendezvous store (not through
   each other directly), and
5. tears down only the children it personally started.

This is why the module docstring says "nothing outside the pods holds test
state." A k8s Job (or a Docker container group) can restart, reschedule, or
be inspected independently, and nothing needs to be told what the plan was —
it just needs the same launch inputs again.

## How the test reaches the runner

`test_deepseek_v2_lite_multi_pod.py` does **not** call any function in
`runner.py` directly. Instead:

- `MULTI_POD_CASES` pairs a scenario id (e.g. `afd-graph-2a2f`) with a layout
  name (e.g. `2pod-role-split` → pod layout string `"2A0F,0A2F"`).
- `build_runner_command()` builds an argv for `python -m
  tests.e2e.multi_pod.runner` — the scenario, the pod layout, a run id, the
  model, output paths, and the shared rendezvous store host — pulling most of
  it from required environment variables (`AFD_E2E_RUN_ID`,
  `AFD_GPU_E2E_MODEL`, `AFD_E2E_GSM8K_OUTPUT`, `AFD_E2E_STORE_HOST`, …).
- `test_multi_pod()` hands that argv to `run_runner()` (`tests/conftest.py`),
  which `subprocess.Popen`s it in a new process group, forwards
  SIGTERM/SIGINT into that group, and raises if it exits non-zero.

Critically, the test file's own docstring spells out the deployment
assumption: **the pods must already exist**. Something else — a human, or
`tests.e2e.multi_pod.driver.k8s` / `driver.docker` — has already created N
pods/containers and is about to (or already did) invoke this same pytest
node once inside each of them. `pytest tests/e2e/models/deepseek_v2_lite/
test_deepseek_v2_lite_multi_pod.py::test_multi_pod[...]` running inside pod 0
and the identical invocation running inside pod 1 together *are* one
multi-pod E2E run. `test_multi_pod()` only supplies the inputs the in-pod
runner cannot infer from its own pod's environment (scenario, layout, store
host); everything pod-specific (index, peer addresses) is resolved by the
runner itself.

So conceptually:

```text
driver (k8s Job / docker driver)
  └─ pod 0: pytest ... test_multi_pod[...]  → runner.py --pod-layout 2A0F,0A2F ...
  └─ pod 1: pytest ... test_multi_pod[...]  → runner.py --pod-layout 2A0F,0A2F ...
```

The two `runner.py` invocations rendezvous with each other over a
`torch.distributed.TCPStore`, not through pytest or the driver.

## `runner.py` code walkthrough

### Imports and constants (lines 1–76)

The runner deliberately imports its "pure" helpers (`identity.py`,
`layout.py`) separately from the rendezvous module, which is the only one
that pulls in `torch.distributed`. The docstring of `identity.py` calls this
out explicitly: identity resolution has to stay unit-testable without a GPU
or a cluster.

The barrier name constants (`ADDRESS_BARRIER`, `LAUNCHED_BARRIER`,
`SERVING_BARRIER`, `VERDICT_BARRIER`) name the four synchronization points
every pod passes through, in order — they show up again in `run_pod()`.
`ROLE_LABELS` is just for readable log prefixes (`ATTN`, `FFN`, `BASELINE`).

### `PodProcess` (lines 79–85)

A tiny frozen dataclass pairing a locally-launched `subprocess.Popen` with
its role and its log-line label. `processes: list[PodProcess]` is threaded
through `main()` → `run_pod()` → `launch_slot()` and is what cleanup and the
liveness checks iterate over.

### `main()` (lines 88–156) — phases 1–5 and the top-level shape

This is the entry point (`if __name__ == "__main__": sys.exit(main())`,
line 442). Reading it top to bottom:

1. **Parse args, configure the scenario, parse the layout.**
   `configure_scenario(args)` (imported from the single-host
   `tests/e2e/runner.py`) is the same function the single-host runner uses —
   it turns a scenario id like `afd-graph-2a2f` into concrete settings
   (`args.num_attention_ranks`, `args.cuda_graph_full_decode_only`,
   `args.afd_connector`, …). Reusing it is what keeps a multi-pod scenario's
   *logical* topology (rank counts, connector, DBO, CUDA graph) identical to
   its single-host counterpart — only pod placement differs.
   `PodLayout.parse(args.pod_layout)` turns `"2A0F,0A2F"` into a tuple of
   `PodSpec(attention=2, ffn=0)` / `PodSpec(attention=0, ffn=2)`.
   `Topology.from_args(args)` captures the scenario's *logical* shape
   (`RoleTopology` per role, connector, `baseline`), independent of how it is
   laid out across pods. `validate_layout()` cross-checks the two: the
   layout's total per-role rank counts must match the topology's, and no
   role's local rank count within a pod may violate that role's TP size
   (a TP group cannot span pods).

2. **Resolve this pod's index.** `resolve_pod_index(layout.num_pods)` (from
   `identity.py`) checks, in order: an explicit `AFD_E2E_POD_INDEX`
   override, the k8s Indexed Job's `JOB_COMPLETION_INDEX`, then falls back to
   the trailing integer of `HOSTNAME` (for hand-applied manifests or the
   Docker driver, which names containers `..-0`, `..-1`, …).

3. **Wait for the rendezvous store's DNS to resolve, then connect.**
   `wait_for_address(args.store_host, args.store_dns_timeout)` exists
   because a freshly-created pod's DNS record can lag its own start by a few
   seconds; without pre-resolving, that shows up as an unexplained
   `TCPStore` connection failure rather than a named "still waiting on DNS"
   state. `Rendezvous(...)` then opens a `TCPStore` — pod 0 is always the
   store's master (`is_master=pod_index == 0`), which is independent of
   which pod *leads* a role or evaluates.

4. **`rendezvous.verify_agreement(layout.canonical(), args.scenario)`.**
   Pod 0 publishes the canonical layout string and scenario id; every other
   pod reads them back and raises immediately if its own values differ. This
   turns a misconfigured deployment (e.g. one pod launched with a different
   `--pod-layout`) into a fast, named failure instead of a barrier timeout
   thirty minutes later.

5. **Resolve every pod's address, plan, and print.**
   `resolve_addresses()` (see below) gives every pod the same ordered list
   of peer addresses. `plan(topology, layout, addresses)[pod_index]` calls
   the pure planning function from `layout.py` and keeps only this pod's
   `PodPlan`. `print_plan()` logs it for the pod's own console/log stream.

6. **Stale-process pre-flight.** `find_stale_run_markers(args.run_id)` scans
   `/proc` for local processes whose `AFD_E2E_RUN_ID` environment variable
   does not match this run's id — a leftover process from a previous run
   still holding a GPU/NPU device or a port. If found, the pod publishes an
   abort (so peers unwind quickly) and raises, rather than letting the
   symptom surface later as an unexplained rendezvous hang.

7. **Run the pod body under `cancellable_run(cleanup)`.**
   `cancellable_run` (from `tests/e2e/cancellation.py`) is shared with the
   single-host runner and gives both the same SIGTERM/SIGINT precedence: a
   signal during the body unwinds immediately into cleanup; a signal that
   arrives *during* cleanup is deferred until cleanup finishes; the
   resulting `SystemExit` then wins over a cleanup error, which in turn wins
   over a body error. `cleanup()` (defined as a closure over `processes`,
   `log_threads`, and `rendezvous`) calls `terminate_processes()` on every
   local child, then joins the log-streaming threads, and best-effort
   publishes its own outcome to the store under `phase/{pod_index}/cleanup`
   (`publish_quietly` — a store failure during teardown must never mask the
   run's actual result). If the body (`run_pod`) raises, `main()` also
   publishes an abort naming which pod failed, so peers still blocked on a
   barrier or on the verdict key unwind within seconds instead of waiting
   out their full timeout.

8. On success, prints a `PASSED` line tagged with this pod's index, scenario,
   and layout, and returns `0`.

### `run_pod()` (lines 159–259) — phases 6–11, the ordered protocol

This is the heart of the coordination logic, and its own comments number the
phases (6 through 11) that continue from `main()`'s 1–5. Every phase either
launches something local or waits on a named rendezvous barrier; **all**
barrier waits pass `on_poll=assert_local_processes_alive`, so a barrier wait
never blocks silently past a local child dying — it notices and raises
within one poll interval.

- **Phases 6–7 — ordered launch.** Which role starts first depends on the
  connector: `uses_async_connector(args)` (true for `CAMAsyncAFDConnector`)
  means Attention hosts the AFD rendezvous and must come up first;
  otherwise FFN does (mirrors `RENDEZVOUS_ROLE_BY_CONNECTOR` in
  `layout.py`). The runner launches this pod's slot for the *first* role
  kind (`launch_slot`, if this pod holds that role), then waits on a
  `f"{first_kind}-launched"` barrier before any pod starts its *second*
  role. This matters because the AFD connector's own rendezvous protocol
  needs the answering side present before the initiating side dials it —
  getting the launch order wrong here would produce connector-level
  timeouts unrelated to this runner's own barriers.

- **Phase 8 — "launched" barrier.** Once both of this pod's local role
  processes (if any) are started, it publishes `phase/{pod_index}/launched`
  and waits for every pod. The code comment is explicit that "launched"
  only means every pod's *children are alive*, not that they are serving —
  a weaker, faster-to-reach checkpoint than readiness.

- **Phase 9 — readiness via `/health`.** Only the pod holding the
  *non-headless* Attention slot is polled for `/health` — and the comment
  explains why that's sufficient: the Attention leader's API server binds
  only after the AFD connector rendezvous completes, which itself requires
  every FFN rank to be present. So one health check transitively proves the
  whole distributed AFD world formed. An FFN engine core never builds an API
  server at all (it enters a busy loop — see
  `afd_plugin/compat/patches/engine_core.py:_run_ffn_busy_loop`), and a
  headless DP slot starts no server by construction, so for both of those
  the phase-8 liveness check was already the honest assertion. After the
  optional health poll, the pod publishes `phase/{pod_index}/serving` and
  waits on the `SERVING_BARRIER`.

- **Phase 10 — exactly one evaluator.** `pod_plan.is_evaluator` is true only
  for the pod holding the Attention role's leader rank (set in
  `layout.plan()`). That one pod runs the accuracy/completion check against
  its **own local** API (`run_completion_evaluation` for the async-CAM
  scenario, `run_gsm8k_evaluation` otherwise — both imported unchanged from
  the single-host `tests/e2e/runner.py`), publishes the verdict string to
  the store (`VERDICT_KEY`), and re-raises on failure after recording
  `"fail: {exc}"` and publishing an abort. Every *other* pod instead calls
  `rendezvous.wait_for_key(VERDICT_KEY, ...)` and blocks until that key
  appears (or a peer aborts).

- **Phase 11 — every pod holds the same verdict.** All pods (evaluator
  included) pass through one more barrier (`VERDICT_BARRIER`) so that a slow
  non-evaluator pod can't be torn down by its driver before it has actually
  read the verdict. A final `assert_local_processes_alive()` catches a child
  that died in the narrow window after the last health/liveness check, and
  the pod raises if its verdict isn't `PASS_VERDICT` — this is what makes
  the process's own exit code (and thus the k8s Job / Docker container exit
  code, and thus `test_multi_pod()`'s subprocess result) reflect pass/fail
  for the whole distributed run, not just this pod's local view.

### `launch_slot()` (lines 262–290)

Builds and starts exactly one role's process for this pod. It picks
`build_baseline_command` or `build_vllm_command(args, role=slot.role,
slot=slot)` — both imported unchanged from the single-host runner — and
passes the `RoleSlot` through so `build_vllm_command` can add the
multi-pod-only flags (`--data-parallel-size-local`,
`--data-parallel-start-rank`, `--data-parallel-address`,
`--data-parallel-rpc-port`, `--headless`) *only* when
`slot.spans_pods` is true. That property is what keeps a one-pod layout
producing the exact same argv, character for character, as the single-host
runner — making a 1-pod layout a genuine control case rather than a
different code path pretending to be one. After starting the process, it is
appended to `processes`, its stdout is piped through `stream_output()` onto
a daemon thread prefixed with a pod/role label, and it is immediately polled
once to fail fast if it exited before even reaching the timeout-based checks
later.

### `wait_for_health()` (lines 293–325)

A small polling loop against `/health` (not `/v1/models`) on the Attention
API port, calling the caller-supplied `on_poll` liveness check on every
iteration (including the first, before the first request) so a dead child is
never masked by an HTTP-level retry. Raises `TimeoutError` naming the last
observed error if the deadline passes.

### `deferred_sigkill_pgids()` (lines 328–335)

Delegates to `uses_npu_async_process_cleanup(args)` (shared with the
single-host runner) to decide whether this pod's FFN process needs the NPU
async teardown allowance — FFN workers using the async CAM connector on NPU
can sit in uninterruptible HCCL teardown for tens of seconds after SIGTERM,
so their process groups get a deferred, longer-patience SIGKILL rather than
being force-killed on the same schedule as everything else. This mirrors
the equivalent logic in the single-host `tests/e2e/runner.py:main()`
verbatim, just scoped to *this pod's* FFN process instead of the whole run's.

### `resolve_addresses()` (lines 338–363)

Three ways to learn every pod's address, tried in order:

1. **Explicit list** — `--pod-addresses` or the `AFD_E2E_POD_ADDRESSES`
   environment variable, comma-separated. Used when the driver already
   knows every pod's address up front.
2. **A template** — `--pod-address-template` with an `{index}` field (e.g. a
   StatefulSet-style DNS name `afd-e2e-abc-{index}.afd-e2e-abc`). This
   *skips* the address-exchange barrier entirely, since the addresses are
   derivable without any coordination.
3. **Rendezvous exchange** — the fallback: this pod publishes its own
   `local_address()` under `addr/{pod_index}`, waits on `ADDRESS_BARRIER`
   for every pod to do the same, then reads all of them back in order.

### `publish_quietly()` and `print_plan()` (lines 366–388)

`publish_quietly` wraps a single `rendezvous.set()` in a `try/except
Exception`, printing rather than raising — used only from `cleanup()`,
where the docstring/comment is direct: teardown's own outcome must never
fail the process, because the exit code the driver observes is what
actually reports the run's result. `print_plan` is pure logging: this pod's
address, its peer list, whether it's the evaluator, and per-slot device/DP
placement — useful for debugging a hung run from pod logs alone.

### `parse_args()` (lines 391–439)

Adds the multi-pod-specific CLI surface on top of
`add_scenario_arguments(parser)` (shared with the single-host runner, which
supplies `--model`, `--scenario`, connector/graph/DBO flags, etc. — but
deliberately *not* device selection, since that's derived differently by
each runner). The multi-pod-only arguments are `--pod-layout`, `--run-id`,
`--store-host` / `--store-port` / `--store-dns-timeout`, the three address
sources for `resolve_addresses()`, `--pod-env` (extra `KEY=VALUE`s merged
into every launched process's environment), and the three timeout knobs
(`--launch-timeout`, `--serving-timeout`, `--verdict-timeout`) that bound
each of the barriers above.

## Where the shared logic actually lives

A recurring theme above: most of what looks like "the test's logic" —
scenario configuration, vLLM command construction, GSM8K/completion
evaluation, environment building, process start/stream/terminate — is
**not** duplicated in `runner.py`. It is imported unchanged from the
single-host `tests/e2e/runner.py`. `runner.py` (this module) only adds what
is genuinely different about running as one pod among several:

- identity resolution (`identity.py`),
- deriving a global plan and reading out one pod's slice of it
  (`layout.py`),
- and cross-pod coordination through a shared store (`rendezvous.py`).

This is also why `layout.RoleSlot` is imported directly into
`tests/e2e/runner.py` (`build_vllm_command(..., slot: RoleSlot | None =
None)`) rather than the multi-pod runner reimplementing command
construction: a `slot=None` call from the single-host runner and a
`slot=<this pod's RoleSlot>` call from here are meant to produce identical
output whenever that slot doesn't actually span pods.

## Provisioning: who creates the pods in the first place

`runner.py` assumes N pods/containers already exist and that it is being
invoked once inside each. That provisioning step is a separate concern,
handled by `tests/e2e/multi_pod/driver/k8s.py` and `driver/docker.py`. Both
drivers describe themselves the same way: their whole job is to render and
apply the deployment (a k8s Indexed Job or a set of sibling Docker
containers), block on the platform's own completion primitive (`kubectl
wait` / a container-exit wait), collect exit codes and logs, and clean up —
they hold no test state and make no test decision. Launch order, readiness,
evaluation, and teardown are entirely owned by the pods running this module,
exactly as walked through above.
