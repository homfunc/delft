#!/usr/bin/env python3
import argparse
import numpy as np
import keras
from keras_crf.crf_ops import crf_decode, crf_log_likelihood

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    D = np.load(args.inputs)
    pot = D['potentials']
    trans = D['transitions']
    lens = D['lengths']
    tags = D['tags']

    # Decode
    dec_tags, dec_scores = crf_decode(pot, lens, trans)
    # Log-likelihood (per-example)
    ll = crf_log_likelihood(pot, tags, lens, trans)

    np.savez(args.out,
             decoded=np.array(dec_tags),
             scores=np.array(dec_scores),
             ll=np.array(ll))
    print(f"Saved KCRF results to {args.out}: dec={np.array(dec_tags).shape}, ll={np.array(ll).shape}")

if __name__ == '__main__':
    main()

