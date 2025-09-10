#!/usr/bin/env python3
import argparse
import os
import h5py
import numpy as np
import keras
from delft.sequenceLabelling import Sequence


def np_from_weight_h5(h5_path: str, suffixes):
    vals = {}
    with h5py.File(h5_path, 'r') as f:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            key = name
            arr = np.array(obj)
            for suf in suffixes:
                if key.endswith(suf):
                    vals[suf] = arr
        f.visititems(lambda n, o: visit(n, o))
    return vals


def to_np(x):
    try:
        return keras.ops.convert_to_numpy(x)
    except Exception:
        try:
            return x.numpy()
        except Exception:
            return np.array(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--legacy', required=True, help='Path to legacy HDF5 weights (e.g., model_weights.hdf5)')
    ap.add_argument('--dump-out', required=True, help='Path to write weights saved from the currently loaded model')
    args = ap.parse_args()

    # Load model in current environment (upgraded)
    seq = Sequence(args.model_name)
    seq.load()
    target_model = getattr(seq.model, 'model', seq.model)

    # Save current model weights as HDF5 for comparison, without modifying weights
    os.makedirs(os.path.dirname(args.dump_out), exist_ok=True)
    target_model.save_weights(args.dump_out)
    print('Saved current model weights to', args.dump_out)

    # Extract CRF transitions from in-memory model
    crf = getattr(target_model, 'crf', None)
    if crf is None:
        raise RuntimeError('CRF layer not found on loaded model')
    trans = getattr(crf, 'trans', None)
    if trans is None:
        trans = getattr(crf, 'transition_params', None)
    if trans is None:
        raise RuntimeError('CRF transitions variable not found on loaded model')
    trans_mem = to_np(trans)

    # Read transitions from legacy and from dump files
    # Common suffix names for transitions across implementations
    suffixes = [
        '/crf/transitions:0',
        '/crf/transitions',
        '/crf/chain_kernel:0',
        '/crf/chain_kernel',
        '/crf/U:0',
        '/crf/U',
        '/transitions:0',
        '/transitions',
        '/chain_kernel:0',
        '/chain_kernel',
        '/U:0',
        '/U',
        # Keras 3 save_weights layout for subclassed layers
        '/crf/vars/0',
        'crf/vars/0',
    ]
    legacy_vals = np_from_weight_h5(args.legacy, suffixes)
    dump_vals = np_from_weight_h5(args.dump_out, suffixes)

    def stats(name, a):
        a = np.asarray(a)
        return f"{name}: shape={a.shape} min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f} std={a.std():.6f} l2={np.linalg.norm(a):.6f}"

    # Pick one transition array from legacy and dump if available
    legacy_trans = None
    for suf in suffixes:
        if suf in legacy_vals:
            legacy_trans = legacy_vals[suf]
            legacy_key = suf
            break
    dump_trans = None
    for suf in suffixes:
        if suf in dump_vals:
            dump_trans = dump_vals[suf]
            dump_key = suf
            break

    print('In-memory transitions ->', stats('mem', trans_mem))
    if legacy_trans is not None:
        print('Legacy transitions (', legacy_key, ') ->', stats('legacy', legacy_trans))
        d = trans_mem - legacy_trans
        print('Diff(mem-legacy): l2', np.linalg.norm(d), 'max_abs', float(np.max(np.abs(d))), 'mean', float(d.mean()), 'std', float(d.std()))
    else:
        print('Legacy transitions not found in', args.legacy)

    if dump_trans is not None:
        print('Saved-H5 transitions (', dump_key, ') ->', stats('dump', dump_trans))
        d2 = dump_trans - trans_mem
        print('Diff(dump-mem): l2', np.linalg.norm(d2), 'max_abs', float(np.max(np.abs(d2))), 'mean', float(d2.mean()), 'std', float(d2.std()))
        if legacy_trans is not None:
            d3 = dump_trans - legacy_trans
            print('Diff(dump-legacy): l2', np.linalg.norm(d3), 'max_abs', float(np.max(np.abs(d3))), 'mean', float(d3.mean()), 'std', float(d3.std()))
    else:
        print('Saved-H5 transitions not found in', args.dump_out)

