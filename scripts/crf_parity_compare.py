#!/usr/bin/env python3
import argparse
import numpy as np

def stats(name, a):
    a = np.asarray(a)
    print(f"{name}: shape={a.shape} min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f} std={a.std():.6f} l2={np.linalg.norm(a):.6f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tfa', required=True)
    p.add_argument('--kcrf', required=True)
    p.add_argument('--inputs', required=False, help='Path to inputs .npz to get lengths for masking decoded comparison')
    args = p.parse_args()

    A = np.load(args.tfa)
    B = np.load(args.kcrf)

    # Decode parity (masked to valid tokens if inputs provided)
    if args.inputs:
        D = np.load(args.inputs)
        lens = D['lengths']
        T = A['decoded'].shape[1]
        mask = np.arange(T)[None, :] < lens[:, None]
        a_dec = A['decoded'][mask]
        b_dec = B['decoded'][mask]
    else:
        a_dec = A['decoded']
        b_dec = B['decoded']
    dec_eq = np.array_equal(a_dec, b_dec)
    print(f"decoded equal? {dec_eq}")
    if not dec_eq:
        mismatch = np.where(a_dec != b_dec)
        print(f"decoded mismatches (masked): {mismatch[0].size}")

    # Scores parity (best-path scores)
    try:
        np.testing.assert_allclose(A['scores'], B['scores'], rtol=1e-6, atol=1e-6)
        print("scores equal (within tol)")
    except AssertionError as e:
        print("scores differ (within tol check failed)")
        stats('scores.tfa', A['scores'])
        stats('scores.kcrf', B['scores'])

    # Log-likelihood parity (per-example)
    try:
        np.testing.assert_allclose(A['ll'], B['ll'], rtol=1e-6, atol=1e-6)
        print("ll equal (within tol)")
    except AssertionError as e:
        print("ll differ (within tol check failed)")
        stats('ll.tfa', A['ll'])
        stats('ll.kcrf', B['ll'])

if __name__ == '__main__':
    main()

