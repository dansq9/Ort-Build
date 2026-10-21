#!/usr/bin/env python3
"""Check that a reduced ONNX Runtime op config covers every operator the release models need.

For each model, as shipped and as the graph ONNX Runtime produces at each optimisation level the
app may use (disable/basic/extended/all), every (domain, op_type) node, including nodes inside
subgraphs, must appear in the config. Prints one line per model and stage, a per-model total and
the union over all models, and lists config entries none of the models needs on this host
(informational: NCHWc/FusedConv appear only on x86 at ALL). Exits non-zero and lists the missing
ops if any model is not covered.

Needs `onnx` and `onnxruntime` (same version as the reduced build) installed:
  python -m pip install onnx onnxruntime==1.31.0
  tools/ort-build/fetch_release_models.sh models/
  python tools/ort-build/check_ops_covered.py tools/ort-build/release_models.required_operators_and_types.config \
      models/scnet.int8.onnx models/nmp.onnx models/spleeter5.fp16w.onnx
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort

LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}

Op = tuple[str, str]


def config_ops(path: Path) -> set[Op]:
    ops: set[Op] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        domain, _opset, spec = line.split(";", 2)
        depth, cur = 0, ""
        for ch in spec + ",":
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if ch == "," and depth == 0:
                ops.add((domain, cur.split("{", 1)[0].strip()))
                cur = ""
            else:
                cur += ch
    return ops


def graph_ops(graph: onnx.GraphProto) -> set[Op]:
    ops: set[Op] = set()
    for node in graph.node:
        ops.add((node.domain or "ai.onnx", node.op_type))
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                ops |= graph_ops(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    ops |= graph_ops(g)
    return ops


def optimized_ops(model: Path, level: ort.GraphOptimizationLevel) -> set[Op]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "optimized.onnx"
        so = ort.SessionOptions()
        so.graph_optimization_level = level
        so.optimized_model_filepath = str(out)
        so.enable_cpu_mem_arena = False  # the app runs with the arena off
        so.log_severity_level = 3  # hide the "NCHWc is hardware specific" warning at ALL
        ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        return graph_ops(onnx.load(str(out), load_external_data=False).graph)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    allowed = config_ops(Path(argv[1]))
    failed = False
    needed_by_any: set[Op] = set()
    for model in map(Path, argv[2:]):
        needed = {"model": graph_ops(onnx.load(str(model), load_external_data=False).graph)}
        for name, level in LEVELS.items():
            needed[f"optimized:{name}"] = optimized_ops(model, level)
        for stage, ops in needed.items():
            missing = sorted(ops - allowed)
            print(f"{model.name} [{stage}] {len(ops)} op types: {'OK' if not missing else f'MISSING {missing}'}")
            failed |= bool(missing)
        union = set().union(*needed.values())
        print(f"{model.name} [any stage] {len(union)} op types: {'OK' if union <= allowed else 'MISSING'}")
        needed_by_any |= union
    print(f"all models [any stage] {len(needed_by_any)} op types: {'OK' if needed_by_any <= allowed else 'MISSING'}")
    unused = sorted(allowed - needed_by_any)
    print(f"onnxruntime {ort.__version__}; config has {len(allowed)} op types"
          + (f"; not seen on this host: {unused}" if unused else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
