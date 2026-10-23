# FFN FaaS P1 mixed-core compilation probe

Experimental one-work primitive for the existing V2-Lite H2048/I1408 BF16
boundary. It groups routes on AIV, runs Cube-only `MatmulImpl` projections,
applies AIV SwiGLU and scatters individual expert outputs. It has no HCCL,
service, scheduler or production registration. After validation, the intended
integration is the existing F-side receive/select/compute/writeback entry.

On the matched aarch64 CANN 9.1 CPU build environment:

```bash
python build_cpu.py --build-dir /private/task/path/build
```

The script limits compilation to two jobs within a cumulative ten-minute
window, reserves time for cleanup and preserves every attempt. Use
`--soc-version Ascend910_9382` for that SKU; the default is `Ascend910_9392`.
The same parameter selects the compiler target and CPU platform/tiling query;
keep each target in a separate build directory and match the device exactly.
It builds a device library and
runs only the CPU platform/Matmul tiling query. The platform's generic library
workspace budget is reported separately from persistent scratch and per-core
API requirements. Device workspace use still requires review and execution.

Compilation does not prove Cube execution, event fanout, visibility or BF16
numerics. Planned device cases include geometry before handshakes, idle/error,
zero owned routes, one/two tokens and one expert with 96 rows (six M tiles,
including rows 16–31). Device execution requires a separate approved bound;
the fixed old-AFD/FaaS performance benchmark remains the final performance test.

Within an approved device window, `run_device.py` takes the compiled `--library`,
400-byte `--tiling`, existing synthetic `--input` fixture and a new
`--output-dir`. Run it with this checkout on `PYTHONPATH`; it reuses the E2E
process-group cleanup helper. Geometry must pass before any compute launch.
The cases share one scratch allocation, and the first mapping, numeric or
deadline failure stops the sequence. Each launch has a 20-second event deadline
inside a worker with a 120-second process deadline. JSON reports and tensors
are preserved for inspection; the resource window and task stop are owned by
the caller.
