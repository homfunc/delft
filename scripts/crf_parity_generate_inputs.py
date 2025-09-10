#!/usr/bin/env python3
import argparse
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--B', type=int, default=8)
    p.add_argument('--T', type=int, default=10)
    p.add_argument('--N', type=int, default=6)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()

    rs = np.random.RandomState(args.seed)
    B, T, N = args.B, args.T, args.N
    potentials = rs.randn(B, T, N).astype('float32')
    transitions = rs.randn(N, N).astype('float32')
    lengths = rs.randint(1, T+1, size=(B,), dtype='int32')
    tags = rs.randint(0, N, size=(B, T), dtype='int32')

    np.savez(args.out,
             potentials=potentials,
             transitions=transitions,
             lengths=lengths,
             tags=tags,
             meta=dict(B=B, T=T, N=N, seed=args.seed))
    print(f"Saved inputs to {args.out}: pot={potentials.shape}, trans={transitions.shape}, len={lengths.shape}, tags={tags.shape}")

if __name__ == '__main__':
    main()

