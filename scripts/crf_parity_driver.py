#!/usr/bin/env python3
import os
import subprocess
import tempfile
import json
import csv
import shutil
from pathlib import Path

UPG_ROOT = Path('/home/m_thing/development/delft')
ORIG_ROOT = Path('/home/m_thing/development/delft-original')

GEN = UPG_ROOT / 'scripts' / 'crf_parity_generate_inputs.py'
RUN_TFA = UPG_ROOT / 'scripts' / 'crf_parity_run_tfa.py'
RUN_KCRF = UPG_ROOT / 'scripts' / 'crf_parity_run_kcrf.py'
CMP = UPG_ROOT / 'scripts' / 'crf_parity_compare.py'

MAMBA_ENV = 'delft-original-py8'
# Where to write summary artifacts
SUMMARY_JSON = Path('/tmp/crf_parity_summary.json')
SUMMARY_CSV = Path('/tmp/crf_parity_summary.csv')

# Resolve micromamba/mamba binary robustly without relying on shell init files
MAMBA_BIN = (
    os.environ.get("MICROMAMBA_EXE")
    or os.environ.get("MAMBA_EXE")
    or shutil.which("micromamba")
    or shutil.which("mamba")
)
if not MAMBA_BIN:
    raise RuntimeError(
        "Could not find micromamba/mamba. Set MICROMAMBA_EXE or MAMBA_EXE, or ensure 'micromamba' or 'mamba' is on PATH."
    )

cases = [
    # (B,T,N,seeds)
    (4, 7, 5, [1, 2, 3]),
    (8, 10, 6, [7, 11]),
    (2, 3, 4, [5, 9]),
    (5, 9, 7, [13]),
]

def run(cmd, env=None, shell=False):
    print('> ' + (cmd if isinstance(cmd, str) else ' '.join(cmd)))
    return subprocess.run(cmd, env=env, shell=shell, check=True, text=True)



def main():
    tmpdir = Path(tempfile.gettempdir())
    total = 0
    passed = 0
    results = []

    for (B, T, N, seeds) in cases:
        for seed in seeds:
            total += 1
            tag = f'B{B}_T{T}_N{N}_seed{seed}'
            inputs = tmpdir / f'crf_inputs_{tag}.npz'
            tfa_out = tmpdir / f'crf_tfa_{tag}.npz'
            kcrf_out = tmpdir / f'crf_kcrf_{tag}.npz'

            # 1) Generate inputs
            run(['python', str(GEN), '--out', str(inputs), '--B', str(B), '--T', str(T), '--N', str(N), '--seed', str(seed)])

            # 2) Run TFA in original env (no shell; no reliance on interactive dotfiles)
            orig_env = os.environ.copy()
            orig_env["PYTHONPATH"] = str(ORIG_ROOT)
            run([MAMBA_BIN, "run", "-n", MAMBA_ENV, "python", str(RUN_TFA), "--inputs", str(inputs), "--out", str(tfa_out)], env=orig_env)

            # 3) Run KCRF in upgraded env
            env = os.environ.copy()
            env.setdefault('KERAS_BACKEND', 'tensorflow')
            run(['python', str(RUN_KCRF), '--inputs', str(inputs), '--out', str(kcrf_out)], env=env)

            # 4) Compare
            cmp_proc = subprocess.run(['python', str(CMP), '--tfa', str(tfa_out), '--kcrf', str(kcrf_out), '--inputs', str(inputs)], capture_output=True, text=True)
            print(cmp_proc.stdout)
            ok = ('decoded equal? True' in cmp_proc.stdout and
                  'scores equal (within tol)' in cmp_proc.stdout and
                  'll equal (within tol)' in cmp_proc.stdout)
            results.append({
                'B': B,
                'T': T,
                'N': N,
                'seed': seed,
                'inputs': str(inputs),
                'tfa_out': str(tfa_out),
                'kcrf_out': str(kcrf_out),
                'stdout': cmp_proc.stdout.strip(),
                'stderr': cmp_proc.stderr.strip(),
                'pass': bool(ok),
            })
            if ok:
                passed += 1
            else:
                print('FAIL:', tag)
                if cmp_proc.stderr:
                    print(cmp_proc.stderr)

    # Write summaries
    with open(SUMMARY_JSON, 'w') as jf:
        json.dump({'passed': passed, 'total': total, 'cases': results}, jf, indent=2)
    with open(SUMMARY_CSV, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['B','T','N','seed','pass','inputs','tfa_out','kcrf_out'])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ['B','T','N','seed','pass','inputs','tfa_out','kcrf_out']})

    print(f'Parity summary: {passed}/{total} cases passed')
    print(f'Wrote: {SUMMARY_JSON}\n       : {SUMMARY_CSV}')

if __name__ == '__main__':
    main()

