"""Verify that each DF3 *.webnn.onnx produces the same output as its original.

Runs both variants on the CPU EP with identical dummy inputs and diffs every
named output. Originals are ground truth. Matches should be within fp32 rounding
(maxAbs < 1e-3).

Usage:
    python scripts/verify_webnn_rewrite.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
DF3_DIR = REPO_ROOT / "packages" / "app" / "studio" / "public" / "models" / "df3"

SEED = 123
SEQ = 50  # moderate sequence; enough to catch recurrence issues


def _build_dummy_inputs_for(session: ort.InferenceSession, rng: np.random.Generator) -> dict[str, np.ndarray]:
    feeds = {}
    for inp in session.get_inputs():
        shape = []
        for d in inp.shape:
            if isinstance(d, int):
                shape.append(d)
            elif d == "S":
                shape.append(SEQ)
            else:
                shape.append(1)
        feeds[inp.name] = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    return feeds


def verify(name: str) -> bool:
    original = DF3_DIR / f"{name}.onnx"
    rewritten = DF3_DIR / f"{name}.webnn.onnx"
    print(f"\n== {name} ==")
    sess_a = ort.InferenceSession(str(original), providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(str(rewritten), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(SEED)
    feeds = _build_dummy_inputs_for(sess_a, rng)
    for k, v in feeds.items():
        print(f"  feed[{k}] shape={list(v.shape)}")
    out_a = sess_a.run(None, feeds)
    out_b = sess_b.run(None, feeds)
    names_a = [o.name for o in sess_a.get_outputs()]
    names_b = [o.name for o in sess_b.get_outputs()]
    if names_a != names_b:
        print(f"  WARNING: output name order differs: {names_a} vs {names_b}")
    ok = True
    # Diff by name (map both to dicts).
    map_a = dict(zip(names_a, out_a))
    map_b = dict(zip(names_b, out_b))
    for key in map_a:
        if key not in map_b:
            print(f"  [{key}] MISSING in rewritten")
            ok = False
            continue
        a = np.asarray(map_a[key], dtype=np.float64)
        b = np.asarray(map_b[key], dtype=np.float64)
        if a.shape != b.shape:
            print(f"  [{key}] SHAPE MISMATCH a={a.shape} b={b.shape}")
            ok = False
            continue
        diff = np.abs(a - b)
        maxAbs = float(diff.max())
        rms = float(np.sqrt((diff ** 2).mean()))
        ref_rms = float(np.sqrt((a ** 2).mean()))
        relRms = rms / max(ref_rms, 1e-12)
        verdict = "OK" if maxAbs < 1e-3 else ("WARN" if maxAbs < 1e-1 else "FAIL")
        if verdict == "FAIL":
            ok = False
        print(f"  [{key}] shape={list(a.shape)} maxAbs={maxAbs:.3e} relRms={relRms:.3e} -> {verdict}")
    return ok


def main():
    all_ok = True
    for name in ("enc", "df_dec", "erb_dec"):
        if not verify(name):
            all_ok = False
    print()
    if all_ok:
        print("ALL MODELS MATCH within fp32 tolerance")
    else:
        print("FAILURES present — see above")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
