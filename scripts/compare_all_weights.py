#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
from typing import Dict, Tuple, List

# Utility to read all datasets from an HDF5 weights file

def read_h5_weights(path: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = np.array(obj)
        f.visititems(lambda n, o: visit(n, o))
    return out


def normalize_key(k: str) -> str:
    # Strip trailing :0, unify separators
    if k.endswith(':0'):
        k = k[:-2]
    return k


def alias_candidates(dump_key: str) -> List[str]:
    # Produce likely legacy suffix candidates for a given Keras 3 save_weights key
    k = normalize_key(dump_key)
    parts = k.split('/')
    cands: List[str] = []
    # Heuristics
    if '/crf/' in k:
        if k.endswith('/vars/0'):
            cands += ['crf/transitions', 'crf/chain_kernel', 'crf/U']
        if k.endswith('/vars/1'):
            cands += ['crf/left_boundary', 'crf/b_start']
        if k.endswith('/vars/2'):
            cands += ['crf/right_boundary', 'crf/b_end']
    # Dense
    if '/dense/' in k and k.endswith('/vars/0'):
        cands += ['dense/kernel']
    if '/dense/' in k and k.endswith('/vars/1'):
        cands += ['dense/bias']
    # Embedding in time_distributed
    if '/time_distributed/' in k and k.endswith('/layer/vars/0'):
        cands += ['char_embeddings/embeddings', 'embeddings']
    # LSTM/GRU cells under bidirectional or time_distributed_1
    if '/cell/' in k:
        if k.endswith('/vars/0'):
            cands += ['kernel']
        if k.endswith('/vars/1'):
            cands += ['recurrent_kernel']
        if k.endswith('/vars/2'):
            cands += ['bias']
    # Allow last token itself as candidate
    cands.append(parts[-1])
    # Also last two tokens
    if len(parts) >= 2:
        cands.append('/'.join(parts[-2:]))
    # And last three
    if len(parts) >= 3:
        cands.append('/'.join(parts[-3:]))
    # Unique preserve order
    seen = set()
    uniq: List[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def match_vars(dump_vars: Dict[str, np.ndarray], legacy_vars: Dict[str, np.ndarray]) -> Tuple[List[Tuple[str,str]], List[str], List[str], Dict[Tuple[int,...], Dict[str, List[str]]]]:
    # Returns: (matches list of (dump_key, legacy_key), ambiguous list of dump_key, missing list of dump_key,
    #           by_shape: mapping shape -> { 'curr': [keys], 'legacy': [keys] } )
    matches: List[Tuple[str,str]] = []
    ambiguous: List[str] = []
    missing: List[str] = []
    legacy_by_shape: Dict[Tuple[int,...], List[str]] = {}
    curr_by_shape: Dict[Tuple[int,...], List[str]] = {}
    for lk, arr in legacy_vars.items():
        legacy_by_shape.setdefault(tuple(arr.shape), []).append(lk)
    for dk, darr in dump_vars.items():
        shape = tuple(darr.shape)
        curr_by_shape.setdefault(shape, []).append(dk)
        candidates = legacy_by_shape.get(shape, [])
        if not candidates:
            missing.append(dk)
            continue
        # Try suffix/alias matching
        aliases = alias_candidates(dk)
        cand_matches = []
        for lk in candidates:
            lk_norm = normalize_key(lk)
            for suf in aliases:
                if lk_norm.endswith(suf) or lk_norm.split('/')[-1] == suf:
                    cand_matches.append(lk)
                    break
        cand_matches = list(dict.fromkeys(cand_matches))
        if len(cand_matches) == 1:
            matches.append((dk, cand_matches[0]))
        elif len(cand_matches) > 1:
            ambiguous.append(dk)
        else:
            # As a fallback, if exactly one candidate contains the same layer name token
            dump_tokens = set(dk.split('/'))
            filtered = [lk for lk in candidates if any(t in lk for t in dump_tokens)]
            if len(filtered) == 1:
                matches.append((dk, filtered[0]))
            else:
                ambiguous.append(dk)
    by_shape: Dict[Tuple[int,...], Dict[str, List[str]]] = {}
    for shape, lst in curr_by_shape.items():
        by_shape.setdefault(shape, {})['curr'] = lst
    for shape, lst in legacy_by_shape.items():
        by_shape.setdefault(shape, {})['legacy'] = lst
    return matches, ambiguous, missing, by_shape


def greedy_match_by_shape(curr_vars: Dict[str, np.ndarray], legacy_vars: Dict[str, np.ndarray], by_shape: Dict[Tuple[int,...], Dict[str, List[str]]], pre_matched: List[Tuple[str,str]]):
    # Build sets of already matched keys
    matched_curr = set(dk for dk, _ in pre_matched)
    matched_legacy = set(lk for _, lk in pre_matched)
    all_matches = list(pre_matched)
    # For each shape group, greedily assign remaining keys by minimal L2 distance
    for shape, groups in by_shape.items():
        curr_keys = [k for k in groups.get('curr', []) if k not in matched_curr]
        legacy_keys = [k for k in groups.get('legacy', []) if k not in matched_legacy]
        if not curr_keys or not legacy_keys:
            continue
        # Build all pairwise distances
        pairs = []
        for dk in curr_keys:
            a = curr_vars[dk]
            for lk in legacy_keys:
                b = legacy_vars[lk]
                if a.shape != b.shape:
                    continue
                d = a - b
                l2 = float(np.linalg.norm(d))
                pairs.append((l2, dk, lk))
        # Greedy selection of minimal pairs
        pairs.sort(key=lambda x: x[0])
        used_curr = set()
        used_leg = set()
        for l2, dk, lk in pairs:
            if dk in used_curr or lk in used_leg:
                continue
            all_matches.append((dk, lk))
            used_curr.add(dk)
            used_leg.add(lk)
            matched_curr.add(dk)
            matched_legacy.add(lk)
            if len(used_curr) == len(curr_keys) or len(used_leg) == len(legacy_keys):
                break
    return all_matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legacy', required=True, help='Path to legacy HDF5 weights (e.g., model_weights.hdf5)')
    ap.add_argument('--current', required=True, help='Path to weights saved from current model via save_weights')
    args = ap.parse_args()

    legacy_vars = read_h5_weights(args.legacy)
    curr_vars = read_h5_weights(args.current)

    matches_name, ambiguous, missing, by_shape = match_vars(curr_vars, legacy_vars)

    print('Total current vars:', len(curr_vars))
    print('Total legacy vars:', len(legacy_vars))
    print('Name/alias matched uniquely:', len(matches_name))
    print('Ambiguous (name heuristic):', len(ambiguous))
    print('No-shape-match:', len(missing))

    # Compute diffs for name/alias matched
    diffs = []
    for dk, lk in matches_name:
        a = curr_vars[dk]
        b = legacy_vars[lk]
        d = a - b
        diffs.append((dk, lk, float(np.linalg.norm(d)), float(np.max(np.abs(d))), float(d.mean()), float(d.std())))
    diffs.sort(key=lambda x: x[2], reverse=True)
    print('\nTop 20 diffs (by L2) for name/alias matches:')
    for row in diffs[:20]:
        dk, lk, l2, mx, mn, sd = row
        print(f"- {dk}  <->  {lk}\n    L2={l2:.6f} max_abs={mx:.6f} mean={mn:.6f} std={sd:.6f}")

    # Now perform greedy per-shape matching to report diffs for all variables
    all_matches = greedy_match_by_shape(curr_vars, legacy_vars, by_shape, matches_name)
    print(f"\nGreedy per-shape matches: {len(all_matches)} (should be <= min(total_current, total_legacy))")
    # Compute diffs for greedy matches
    diffs_all = []
    zeros = 0
    for dk, lk in all_matches:
        a = curr_vars[dk]
        b = legacy_vars[lk]
        d = a - b
        l2 = float(np.linalg.norm(d))
        mx = float(np.max(np.abs(d)))
        mn = float(d.mean())
        sd = float(d.std())
        if l2 == 0.0 and mx == 0.0:
            zeros += 1
        diffs_all.append((dk, lk, l2, mx, mn, sd))
    diffs_all.sort(key=lambda x: x[2], reverse=True)
    print(f"Exact zero-diff pairs: {zeros}/{len(all_matches)}")
    print('\nTop 20 diffs (by L2) after greedy per-shape matching:')
    for row in diffs_all[:20]:
        dk, lk, l2, mx, mn, sd = row
        print(f"- {dk}  <->  {lk}\n    L2={l2:.6f} max_abs={mx:.6f} mean={mn:.6f} std={sd:.6f}")

    # CRF-specific if present
    crf_curr = [k for k in curr_vars if '/crf/' in k]
    crf_leg = [k for k in legacy_vars if '/crf/' in k or k.endswith('/U') or k.endswith('/chain_kernel') or k.endswith('/transitions')]
    print(f"\nCRF vars current: {len(crf_curr)}; legacy: {len(crf_leg)}")
    for k in crf_curr:
        print(' current:', k, curr_vars[k].shape)
    for k in crf_leg:
        print(' legacy :', k, legacy_vars[k].shape)

    # Summaries for ambiguous/missing
    if ambiguous:
        print('\nAmbiguous by name (pre-greedy):', len(ambiguous))
        for k in ambiguous[:20]:
            print('  ', k, curr_vars[k].shape)
    if missing:
        print('\nNo legacy var with same shape for current var:', len(missing))
        for k in missing[:20]:
            print('  ', k, curr_vars[k].shape)

if __name__ == '__main__':
    main()

