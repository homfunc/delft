#!/usr/bin/env python3
import argparse
import os
import sys
import keras

from delft.sequenceLabelling import Sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', required=True, help='Model name folder under data/models/sequenceLabelling')
    ap.add_argument('--legacy', required=True, help='Path to legacy HDF5 weights (e.g., model_weights.hdf5)')
    ap.add_argument('--out-weights', required=True, help='Path to write mapped weights (Keras 3 save_weights format, must end with .weights.h5)')
    ap.add_argument('--save-keras', default=None, help='Optional path to write a .keras model after mapping (will not overwrite existing unless you point to a new file)')
    args = ap.parse_args()

    # Load upgraded model (will load model.keras if present)
    seq = Sequence(args.model_name)
    seq.load()
    model = getattr(seq.model, 'model', seq.model)

    # Ensure variables are created for subclassed model
    try:
        build_cfg_fn = getattr(model, 'get_build_config', None)
        build_from_cfg_fn = getattr(model, 'build_from_config', None)
        if callable(build_cfg_fn) and callable(build_from_cfg_fn):
            cfg = build_cfg_fn()
            build_from_cfg_fn(cfg)
    except Exception as be:
        print(f"Warning: symbolic build before mapping failed: {be}")

    # Assign from legacy HDF5 by name/alias
    from delft.utilities.weights import load_weights_by_name_from_h5
    assigned, missing = load_weights_by_name_from_h5(model, args.legacy, verbose=True)
    print(f"Assigned {assigned} variables; missing {len(missing)}")

    # Save mapped weights to an external file for verification and future loading
    os.makedirs(os.path.dirname(args.out_weights), exist_ok=True)
    model.save_weights(args.out_weights)
    print('Mapped weights saved to', args.out_weights)

    # Optionally save a new .keras file (does not overwrite unless specified)
    if args.save_keras:
        try:
            model.save(args.save_keras)
            print('Mapped model saved to', args.save_keras)
        except Exception as e:
            print('Warning: could not save mapped .keras model:', e)


if __name__ == '__main__':
    main()

