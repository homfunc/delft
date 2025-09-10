# -*- coding: utf-8 -*-

import os
import argparse
import fnmatch

from delft.sequenceLabelling.wrapper import Sequence


def convert_model(model_name: str, root: str, force: bool = False) -> str:
    """Attempt to convert one model directory. Returns a status string.
    Status values: converted, already, no_legacy, failed, skipped
    """
    model_dir = os.path.join(root, model_name)
    if not os.path.isdir(model_dir):
        print(f"Skip {model_name}: not a directory")
        return 'skipped'

    keras_path = os.path.join(model_dir, 'model.keras')
    h5_path = os.path.join(model_dir, 'model_weights.hdf5')

    if os.path.exists(keras_path) and not force:
        print(f"Already converted: {model_name}")
        return 'already'

    if not os.path.exists(h5_path) and not force:
        print(f"No legacy weights for: {model_name}")
        return 'no_legacy'

    # Trigger load which performs auto-convert in BaseModel.load
    seq = Sequence(model_name)
    try:
        # Skip embedding initialization; use embedding size stored in the saved config
        seq.load(dir_path=root, skip_embeddings=True)
        print(f"Converted {model_name} -> model.keras")
        return 'converted'
    except Exception as e:
        print(f"Failed to convert {model_name}: {e}")
        return 'failed'


def _matches(name: str, includes: list[str], excludes: list[str]) -> bool:
    if includes:
        ok = any(fnmatch.fnmatch(name, pat) for pat in includes)
        if not ok:
            return False
    if excludes:
        if any(fnmatch.fnmatch(name, pat) for pat in excludes):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert legacy DeLFT sequence models from HDF5 weights to Keras 3 .keras format")
    parser.add_argument('--root', default='data/models/sequenceLabelling/', help='Root directory containing model subdirectories')
    parser.add_argument('--model', default=None, help='Specific model name to convert (subdirectory name)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing .keras file if present')
    parser.add_argument('--include', action='append', default=[], help='Glob pattern(s) to include (can be repeated)')
    parser.add_argument('--exclude', action='append', default=[], help='Glob pattern(s) to exclude (can be repeated)')

    args = parser.parse_args()

    root = args.root

    stats = {
        'converted': [],
        'already': [],
        'no_legacy': [],
        'failed': [],
        'skipped': []
    }

    if args.model:
        status = convert_model(args.model, root, force=args.force)
        stats[status].append(args.model)
    else:
        # Convert all immediate subdirectories under root
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            if not _matches(name, args.include, args.exclude):
                stats['skipped'].append(name)
                continue
            status = convert_model(name, root, force=args.force)
            stats[status].append(name)

    # Summary report
    total = sum(len(v) for v in stats.values())
    print("\n===== Conversion summary =====")
    print(f"Total processed: {total}")
    for key in ['converted', 'already', 'no_legacy', 'skipped', 'failed']:
        print(f"{key:>10}: {len(stats[key])}")
    if stats['failed']:
        print("Failed models:")
        for n in stats['failed']:
            print("  -", n)


if __name__ == '__main__':
    main()

