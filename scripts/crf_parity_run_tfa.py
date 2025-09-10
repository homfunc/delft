#!/usr/bin/env python3
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    D = np.load(args.inputs)
    pot = tf.convert_to_tensor(D['potentials'])
    trans = tf.convert_to_tensor(D['transitions'])
    lens = tf.convert_to_tensor(D['lengths'])
    tags = tf.convert_to_tensor(D['tags'])

    # Decode
    dec_tags, dec_scores = tfa.text.crf_decode(pot, trans, lens)
    # Log-likelihood (per-example)
    ll, _ = tfa.text.crf_log_likelihood(inputs=pot, tag_indices=tags, sequence_lengths=lens, transition_params=trans)

    np.savez(args.out,
             decoded=dec_tags.numpy(),
             scores=dec_scores.numpy(),
             ll=ll.numpy())
    print(f"Saved TFA results to {args.out}: dec={dec_tags.shape}, ll={ll.shape}")

if __name__ == '__main__':
    main()

