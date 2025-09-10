#!/usr/bin/env python3
import argparse
import numpy as np
import keras
from keras_crf.crf_ops import crf_decode

def stats(name, arr):
    a = np.asarray(arr)
    print(f"{name}: shape={a.shape} dtype={a.dtype} min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f} std={a.std():.6f} l2={np.linalg.norm(a):.6f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='.npz from repo A')
    ap.add_argument('--b', required=True, help='.npz from repo B')
    args = ap.parse_args()

    A = np.load(args.a)
    B = np.load(args.b)

    # Print transition and potential stats
    stats('A.transitions', A['transitions'])
    stats('B.transitions', B['transitions'])
    stats('A.potentials', A['potentials'])
    stats('B.potentials', B['potentials'])

    def decode_pack(pack):
        pot = pack['potentials']
        ln = pack['lengths']
        tr = pack['transitions']
        # Use backend-agnostic decode; expect (tags, scores)
        tags, scores = crf_decode(pot, ln, tr)
        return np.array(tags), np.array(scores)

    # 1) Sanity: decode(A.potentials) equals A.decoded saved from the model
    tags_A, scores_A = decode_pack(A)
    model_dec_A = A['decoded']
    eq_A = np.array_equal(tags_A, model_dec_A)

    tags_B, scores_B = decode_pack(B)
    model_dec_B = B['decoded']
    eq_B = np.array_equal(tags_B, model_dec_B)

    print(f"A: decode(potentials) == model_decoded ? {eq_A}  (shape {tags_A.shape})")
    if not eq_A:
        disagree = np.where(tags_A != model_dec_A)
        print(f"  A model vs recompute disagree at positions: count={disagree[0].size}")
    print(f"B: decode(potentials) == model_decoded ? {eq_B}  (shape {tags_B.shape})")
    if not eq_B:
        disagree = np.where(tags_B != model_dec_B)
        print(f"  B model vs recompute disagree at positions: count={disagree[0].size}")

    # 2) Cross-repo transitions sanity (same shapes; closeness summary)
    tr_A = A['transitions']
    tr_B = B['transitions']
    if tr_A.shape == tr_B.shape:
        diff = np.linalg.norm(tr_A - tr_B)
        print(f"Transitions L2 diff: {diff:.6f} (shape {tr_A.shape})")
    else:
        print(f"Transitions shape mismatch: {tr_A.shape} vs {tr_B.shape}")
    # Also compare masks/first/last if present
    for key in ['first_idx', 'last_idx']:
        if key in A and key in B:
            if A[key].shape == B[key].shape:
                d = np.linalg.norm(A[key] - B[key])
                print(f"{key} L2 diff: {d:.6f}")
    if 'mask' in A and 'mask' in B:
        if A['mask'].shape == B['mask'].shape:
            equal_mask = np.array_equal(A['mask'], B['mask'])
            print(f"mask equal? {equal_mask}")

    # 3) If we plug A's transitions and B's potentials (or vice versa), decode deterministically
    #    (not a correctness test across repos, but demonstrates crf_decode consistency)
    if A['potentials'].shape == B['potentials'].shape and np.array_equal(A['lengths'], B['lengths']) and tr_A.shape == tr_B.shape:
        tags_mix1, _ = crf_decode(A['potentials'], A['lengths'], B['transitions'])
        tags_mix2, _ = crf_decode(B['potentials'], B['lengths'], A['transitions'])
        print(f"Mixed decode shapes: {np.array(tags_mix1).shape}, {np.array(tags_mix2).shape}")

    # 4) Report A vs B decoded agreement when each uses its own inputs (this reflects upstream features)
    if tags_A.shape == tags_B.shape:
        total = tags_A.size
        agree = int((tags_A == tags_B).sum())
        print(f"A vs B decoded agreement: {agree}/{total} = {agree/total:.4f}")

if __name__ == '__main__':
    main()

