"""Runs each model on the stock onnxruntime and on the reduced build (two venvs) with the same seeded
input, at every graph optimisation level, and fails if outputs differ beyond tolerance.

    python parity.py dump  OUT.npz MODEL...   # run in a venv, writes outputs
    python parity.py compare A.npz B.npz      # stock vs reduced
"""
import sys
import numpy as np


def inputs_for(sess):
    rng = np.random.default_rng(2026)
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        x = rng.standard_normal(shape).astype(np.float32)
        feeds[i.name] = np.abs(x) if "mag" in i.name else x
    return feeds


def dump(out, models):
    import onnxruntime as ort
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    result = {}
    for model in models:
        for name, level in levels.items():
            so = ort.SessionOptions()
            so.graph_optimization_level = level
            sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
            for k, v in enumerate(sess.run(None, inputs_for(sess))):
                result[f"{model.split('/')[-1]}|{name}|{k}"] = v
    np.savez(out, **result)
    print(f"{ort.__version__}: {len(result)} outputs")


def compare(a, b):
    a, b = np.load(a), np.load(b)
    worst = 0.0
    assert set(a.files) == set(b.files), "different outputs"
    for k in sorted(a.files):
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        rel = np.max(np.abs(x - y)) / (np.max(np.abs(x)) + 1e-12)
        worst = max(worst, rel)
        print(f"{k}: max rel diff {rel:.2e}")
    print(f"worst {worst:.2e}")
    sys.exit(0 if worst < 1e-4 else 1)


if __name__ == "__main__":
    {"dump": lambda: dump(sys.argv[2], sys.argv[3:]), "compare": lambda: compare(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
