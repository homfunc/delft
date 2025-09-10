#!/usr/bin/env python3
import argparse
import numpy as np
import keras
from delft.sequenceLabelling import Sequence

def stats(name, a):
    a = np.asarray(a)
    print(f"{name}: shape={a.shape} min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f} std={a.std():.6f} l2={np.linalg.norm(a):.6f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    seq = Sequence(args.model_name)
    seq.load()
    inner = getattr(seq.model, 'model', seq.model)

    # Access CRF transitions
    crf = getattr(inner, 'crf', None)
    if crf is None:
        raise RuntimeError('CRF layer not found on inner model')
    trans = crf.trans if hasattr(crf, 'trans') else getattr(crf, 'transition_params', None)
    if trans is None:
        raise RuntimeError('CRF transitions variable not found')

    # Materialize as numpy
    try:
        trans_np = keras.ops.convert_to_numpy(trans)
    except Exception:
        try:
            trans_np = trans.numpy()
        except Exception:
            trans_np = np.array(trans)

    np.save(args.out, trans_np)
    stats('transitions', trans_np)
    print('saved transitions to', args.out)

