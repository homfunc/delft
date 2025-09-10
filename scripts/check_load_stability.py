#!/usr/bin/env python3
import argparse
import os
import tempfile
import numpy as np
import keras
from delft.sequenceLabelling import Sequence


def save_weights(model_name: str, out_path: str):
    seq = Sequence(model_name)
    seq.load()
    target = getattr(seq.model, 'model', seq.model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    target.save_weights(out_path)
    return out_path


def compare_two_h5(a_path: str, b_path: str):
    import h5py
    def read(path):
        d = {}
        with h5py.File(path, 'r') as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    d[name] = np.array(obj)
            f.visititems(lambda n, o: visit(n, o))
        return d
    A = read(a_path)
    B = read(b_path)
    keys = sorted(set(A.keys()) & set(B.keys()))
    # if Keras 3, main CRF vars are under crf/vars/*, match by identical keys
    diffs = []
    for k in keys:
        a = A[k]
        b = B[k]
        if a.shape != b.shape:
            continue
        d = a - b
        l2 = float(np.linalg.norm(d))
        mx = float(np.max(np.abs(d)))
        if l2 > 0 or mx > 0:
            diffs.append((k, l2, mx))
    return diffs, len(A), len(B)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    # First load/save
    path1 = os.path.join(args.outdir, 'load1.weights.h5')
    save_weights(args.model_name, path1)
    # Clear Keras session between loads to avoid any residual state
    try:
        keras.backend.clear_session()
    except Exception:
        pass
    # Second load/save
    path2 = os.path.join(args.outdir, 'load2.weights.h5')
    save_weights(args.model_name, path2)

    diffs, nA, nB = compare_two_h5(path1, path2)
    print('Saved files:', path1, path2)
    print('Dataset counts:', nA, nB)
    if not diffs:
        print('OK: Weights stable across two loads (identical tensors)')
    else:
        print('WARNING: Found differences across loads:', len(diffs))
        for k, l2, mx in diffs[:20]:
            print(f'- {k}: L2={l2:.6f} max_abs={mx:.6f}')

