#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime


def run_cmd(cmd: str, cwd: str = None) -> int:
    print("\n>>>", cmd)
    proc = subprocess.run(cmd, shell=True, cwd=cwd)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Run original vs upgraded DeLFT parity dumps and comparisons")
    # Resolve mamba/micromamba/conda runner
    import shutil
    mamba_bin = os.environ.get('MAMBA_BIN')
    if not mamba_bin:
        # Prefer micromamba if available; otherwise mamba; otherwise conda
        for cand in ('micromamba', 'mamba', 'conda'):
            if shutil.which(cand):
                mamba_bin = cand
                break
    if not mamba_bin:
        print('ERROR: Could not find micromamba/mamba/conda on PATH. Please set MAMBA_BIN.')
        sys.exit(2)
    ap.add_argument('--orig-env', required=True, help='Conda env name for delft-original (e.g., delft-original-py8)')
    ap.add_argument('--orig-root', required=True, help='Path to delft-original repo root')
    ap.add_argument('--upgr-env', required=True, help='Conda env name for upgraded DeLFT (e.g., kaggle)')
    ap.add_argument('--upgr-root', required=True, help='Path to upgraded DeLFT repo root')
    ap.add_argument('--model-name', required=True, help='Model name (e.g., grobid-date-BidLSTM_CRF)')
    ap.add_argument('--orig-input', required=True, help='Path to CRF-format input file in delft-original repo')
    ap.add_argument('--upgr-input', required=True, help='Path to CRF-format input file in upgraded repo')
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--out-dir', default='/tmp/parity')
    ap.add_argument('--skip-crf', action='store_true', help='Skip CRF IO dumps')
    ap.add_argument('--skip-base', action='store_true', help='Skip base IO dumps/comparison')
    ap.add_argument('--no-delete-keras', action='store_true', help='Do not delete model.keras before runs')
    args = ap.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.out_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    model_dir_orig = os.path.join(args.orig_root, 'data/models/sequenceLabelling', args.model_name)
    model_dir_upgr = os.path.join(args.upgr_root, 'data/models/sequenceLabelling', args.model_name)

    keras_orig = os.path.join(model_dir_orig, 'model.keras')
    keras_upgr = os.path.join(model_dir_upgr, 'model.keras')

    if not args.no_delete_keras:
        # Ensure we only load legacy HDF5
        if os.path.exists(keras_orig):
            print(f"Removing {keras_orig}")
            try:
                os.remove(keras_orig)
            except Exception as e:
                print(f"Warning: could not remove {keras_orig}: {e}")
        if os.path.exists(keras_upgr):
            print(f"Removing {keras_upgr}")
            try:
                os.remove(keras_upgr)
            except Exception as e:
                print(f"Warning: could not remove {keras_upgr}: {e}")

    # Build common flags
    det_flag = '--deterministic' if args.deterministic else ''
    lim_flag = f"--limit {args.limit}"

    # Paths for outputs
    crf_orig_out = os.path.join(out_dir, 'crf_original.npz')
    crf_upgr_out = os.path.join(out_dir, 'crf_upgraded.npz')
    base_orig_out = os.path.join(out_dir, 'base_original.npz')
    base_upgr_out = os.path.join(out_dir, 'base_upgraded.npz')

    # Dump CRF IO (optional)
    if not args.skip_crf:
        # original
        cmd = (
            f"{shlex.quote(mamba_bin)} run -n {shlex.quote(args.orig_env)} env "
            f"PYTHONPATH={shlex.quote(args.orig_root)} "
            f"KERAS_BACKEND=tensorflow "
            f"python {shlex.quote(os.path.join(args.orig_root, 'scripts/dump_crf_io.py'))} "
            f"--model-name {shlex.quote(args.model_name)} "
            f"--input {shlex.quote(args.orig_input)} "
            f"--out {shlex.quote(crf_orig_out)} {lim_flag} {det_flag}"
        )
        if run_cmd(cmd, cwd=args.orig_root) != 0:
            sys.exit(1)
        # upgraded
        cmd = (
            f"{shlex.quote(mamba_bin)} run -n {shlex.quote(args.upgr_env)} env "
            f"PYTHONPATH={shlex.quote(args.upgr_root)} "
            f"KERAS_BACKEND=tensorflow "
            f"python {shlex.quote(os.path.join(args.upgr_root, 'scripts/dump_crf_io.py'))} "
            f"--model-name {shlex.quote(args.model_name)} "
            f"--input {shlex.quote(args.upgr_input)} "
            f"--out {shlex.quote(crf_upgr_out)} {lim_flag} {det_flag}"
        )
        if run_cmd(cmd, cwd=args.upgr_root) != 0:
            sys.exit(1)

    # Dump Base IO (pre-CRF) and compare (optional)
    if not args.skip_base:
        # original
        cmd = (
            f"{shlex.quote(mamba_bin)} run -n {shlex.quote(args.orig_env)} env "
            f"PYTHONPATH={shlex.quote(args.orig_root)} "
            f"KERAS_BACKEND=tensorflow "
            f"python {shlex.quote(os.path.join(args.orig_root, 'scripts/dump_base_io.py'))} "
            f"--model-name {shlex.quote(args.model_name)} "
            f"--input {shlex.quote(args.orig_input)} "
            f"--out {shlex.quote(base_orig_out)} {lim_flag} {det_flag}"
        )
        if run_cmd(cmd, cwd=args.orig_root) != 0:
            sys.exit(1)
        # upgraded
        cmd = (
            f"{shlex.quote(mamba_bin)} run -n {shlex.quote(args.upgr_env)} env "
            f"PYTHONPATH={shlex.quote(args.upgr_root)} "
            f"KERAS_BACKEND=tensorflow "
            f"python {shlex.quote(os.path.join(args.upgr_root, 'scripts/dump_base_io.py'))} "
            f"--model-name {shlex.quote(args.model_name)} "
            f"--input {shlex.quote(args.upgr_input)} "
            f"--out {shlex.quote(base_upgr_out)} {lim_flag} {det_flag}"
        )
        if run_cmd(cmd, cwd=args.upgr_root) != 0:
            sys.exit(1)
        # compare using the upgraded repo's compare script
        cmd = (
            f"python {shlex.quote(os.path.join(args.upgr_root, 'scripts/compare_base_io.py'))} "
            f"--orig {shlex.quote(base_orig_out)} --upgr {shlex.quote(base_upgr_out)}"
        )
        if run_cmd(cmd, cwd=args.upgr_root) != 0:
            sys.exit(1)

    print("\nAll parity steps completed. Artifacts in:", out_dir)


if __name__ == '__main__':
    main()

