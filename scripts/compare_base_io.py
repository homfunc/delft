#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np


def compare_stage(name: str, a: np.ndarray, b: np.ndarray, lengths: np.ndarray, atol: float = 1e-5):
    if a.shape[0] != b.shape[0]:
        print(f"[WARN] batch mismatch for {name}: {a.shape[0]} vs {b.shape[0]}")
    B = min(a.shape[0], b.shape[0], lengths.shape[0])
    mismatches = 0
    max_abs = 0.0
    for i in range(B):
        L = int(lengths[i])
        aa = a[i, :L]
        bb = b[i, :L]
        if aa.shape != bb.shape:
            print(f"[WARN] {name} shape mismatch at {i}: {aa.shape} vs {bb.shape}")
            mismatches += 1
            continue
        d = float(np.max(np.abs(aa - bb))) if aa.size else 0.0
        max_abs = max(max_abs, d)
        if not np.allclose(aa, bb, atol=atol):
            mismatches += 1
    if a.shape != b.shape:
        print(f"[WARN] {name} shape mismatch: {a.shape} vs {b.shape}")
    print(f"{name}: mismatches={mismatches}/{B} max_abs_diff={max_abs:.6g}")


def compare_equal(name: str, a: np.ndarray, b: np.ndarray):
    eq = np.array_equal(a, b)
    print(f"{name}: equal={eq}")
    return eq


def compare_array(name: str, a: np.ndarray, b: np.ndarray, atol: float = 1e-6):
    if a.shape != b.shape:
        print(f"{name}: shape mismatch {a.shape} vs {b.shape}")
        return False
    max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
    ok = np.allclose(a, b, atol=atol)
    print(f"{name}: allclose={ok} max_abs_diff={max_abs:.6g}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig', required=True, help='Path to original dump_base_io .npz')
    ap.add_argument('--upgr', required=True, help='Path to upgraded dump_base_io .npz')
    ap.add_argument('--atol', type=float, default=1e-5)
    args = ap.parse_args()

    A1 = np.load(args.orig)
    A2 = np.load(args.upgr)

    # lengths should match; if not, bail out
    len1 = A1['lengths']
    len2 = A2['lengths']
    if not np.array_equal(len1, len2):
        print('ERROR: lengths differ between archives')
        print('orig:', len1)
        print('upgr:', len2)
        raise SystemExit(2)

    # Compare tokens/labels JSON if available (paths derived from NPZ path)
    def load_tokens_json(npz_path: str):
        jpath = npz_path + '.tokens.json'
        if os.path.exists(jpath):
            try:
                with open(jpath, 'r', encoding='utf-8') as jf:
                    return json.load(jf)
            except Exception as e:
                print(f"[WARN] could not load {jpath}: {e}")
        return None

    tok1 = load_tokens_json(args.orig)
    tok2 = load_tokens_json(args.upgr)
    if tok1 is not None and tok2 is not None:
        t1 = tok1.get('tokens')
        t2 = tok2.get('tokens')
        if t1 is None or t2 is None:
            print('[WARN] tokens missing in tokens.json files')
        else:
            if len(t1) != len(t2):
                print(f"tokens length mismatch: {len(t1)} vs {len(t2)}")
            else:
                mism = []
                for i, (a, b) in enumerate(zip(t1, t2)):
                    if a != b:
                        mism.append(i)
                if mism:
                    print(f"tokens differ at indices: {mism[:10]}{'...' if len(mism)>10 else ''}")
                else:
                    print('tokens: equal')
        l1 = tok1.get('labels')
        l2 = tok2.get('labels')
        if l1 is not None and l2 is not None:
            if len(l1) != len(l2):
                print(f"labels length mismatch: {len(l1)} vs {len(l2)}")
            else:
                lmism = []
                for i, (a, b) in enumerate(zip(l1, l2)):
                    if a != b:
                        lmism.append(i)
                if lmism:
                    print(f"labels differ at indices: {lmism[:10]}{'...' if len(lmism)>10 else ''}")
                else:
                    print('labels: equal')

    # Compare char-centric inputs when available
    if 'char_ids' in A1.files and 'char_ids' in A2.files:
        c1 = A1['char_ids']
        c2 = A2['char_ids']
        if c1.shape != c2.shape:
            print('char_ids shape mismatch:', c1.shape, c2.shape)
            char_ids_shapes_mismatch = True
        else:
            char_ids_shapes_mismatch = False
        same_char_ids = compare_equal('char_ids', A1['char_ids'], A2['char_ids'])
        if not same_char_ids and not char_ids_shapes_mismatch:
            # Print detailed char-level diffs for first few mismatches
            B, T, C = c1.shape
            shown = 0
            max_show = 20
            print('char-level differences (up to first 20):')
            for i in range(B):
                for t in range(T):
                    print(f"  shown={shown}")
                    # skip if entire token char ids equal
                    if np.array_equal(c1[i, t], c2[i, t]):
                        continue
                    # derive token string if available
                    token_str = None
                    if tok1 is not None and tok1.get('tokens') and i < len(tok1['tokens']) and t < len(tok1['tokens'][i]):
                        token_str = tok1['tokens'][i][t]
                    # compare per char position
                    for ch in range(C):
                        id1 = int(c1[i, t, ch])
                        id2 = int(c2[i, t, ch])
                        print(f"  sample={i} token={t} char_pos={ch} orig_id={id1} upgr_id={id2}")
                        if id1 != id2:
                            ch_char = None
                            ch_ord = None
                            if token_str is not None and ch < len(token_str):
                                ch_char = token_str[ch]
                                try:
                                    ch_ord = ord(ch_char)
                                except Exception:
                                    ch_ord = None
                            print(f"  sample={i} token={t} char_pos={ch} char={repr(ch_char)} ord={ch_ord} orig_id={id1} upgr_id={id2}")
                            shown += 1
                            if shown >= max_show:
                                break
                    if shown >= max_show:
                        break
                if shown >= max_show:
                    break
    if 'char_lengths' in A1.files and 'char_lengths' in A2.files:
        compare_equal('char_lengths', A1['char_lengths'], A2['char_lengths'])
    if 'char_emb' in A1.files and 'char_emb' in A2.files:
        compare_array('char_emb', A1['char_emb'], A2['char_emb'], atol=1e-6)

    # Compare any time_distributed_1 weights captured via path-based keys
    td1_keys_1 = sorted([k for k in A1.files if k.startswith('w__') and '__time_distributed_1__' in k])
    td1_keys_2 = sorted([k for k in A2.files if k.startswith('w__') and '__time_distributed_1__' in k])
    if td1_keys_1 or td1_keys_2:
        common = sorted(set(td1_keys_1).intersection(td1_keys_2))
        missing1 = sorted(set(td1_keys_2) - set(td1_keys_1))
        missing2 = sorted(set(td1_keys_1) - set(td1_keys_2))
        if missing1:
            print('TD1 weights missing in original:', missing1)
        if missing2:
            print('TD1 weights missing in upgraded:', missing2)
        for k in common:
            compare_array(f'TD1 {k}', A1[k], A2[k], atol=1e-6)

    # Compare deep-extracted TD1 LSTM weights when present
    for key in ['td1_fwd_kernel', 'td1_fwd_recurrent_kernel', 'td1_fwd_bias',
                'td1_bwd_kernel', 'td1_bwd_recurrent_kernel', 'td1_bwd_bias']:
        if key in A1.files and key in A2.files:
            compare_array(key, A1[key], A2[key], atol=1e-6)

    # Ensure stage keys exist
    for key in ['A', 'B', 'C', 'D']:
        if key not in A1.files or key not in A2.files:
            print(f'ERROR: missing {key} in one of the archives')
            raise SystemExit(2)

    print('Comparing base-model intermediate tensors (length-aligned) ...')
    if 'E' in A1.files and 'E' in A2.files and 'char_mask' in A1.files and 'char_mask' in A2.files:
        # Compare char embedding outputs only on valid positions (time and char masks)
        E1 = A1['E']
        E2 = A2['E']
        cm1 = A1['char_mask']  # (B,T,max_char), 1 for valid
        cm2 = A2['char_mask']
        # Time mask from lengths
        Bsz = E1.shape[0]
        T = E1.shape[1]
        tmask = (np.arange(T)[None, :] < len1[:, None])  # (B,T)
        # Broadcast to char axis
        tmask3 = tmask[..., None]
        valid1 = (cm1.astype(bool) & tmask3)
        valid2 = (cm2.astype(bool) & tmask3)
        # Only compare where both sides are valid
        valid = (valid1 & valid2)
        if E1.shape != E2.shape or valid.shape != E1[..., 0].shape:
            print('E(char TD Embedding) shape mismatch; skipping masked compare')
        else:
            # If no valid positions (unlikely), skip
            if valid.any():
                diff = np.abs(E1 - E2)
                max_abs = float(diff[valid].max())
                ok = np.allclose(E1[valid], E2[valid], atol=args.atol)
                print(f"E(char TD Embedding): allclose={ok} max_abs_diff={max_abs:.6g}")
            else:
                print('E(char TD Embedding): no valid positions to compare')
    compare_stage('A(chars TD BiLSTM)', A1['A'], A2['A'], len1, atol=args.atol)
    compare_stage('B(concat)', A1['B'], A2['B'], len1, atol=args.atol)
    compare_stage('C(word BiLSTM)', A1['C'], A2['C'], len1, atol=args.atol)
    compare_stage('D(pre-CRF Dense)', A1['D'], A2['D'], len1, atol=args.atol)
