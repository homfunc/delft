#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict

def load_preds(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            items.append(obj)
    return items

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--upgraded', required=True)
    ap.add_argument('--original', required=True)
    args = ap.parse_args()

    u = load_preds(args.upgraded)
    o = load_preds(args.original)
    assert len(u) == len(o), f'Length mismatch: {len(u)} vs {len(o)}'

    total = 0
    agree = 0
    by_true = Counter()
    conf = defaultdict(Counter)
    sample_diffs = []

    for uu, oo in zip(u, o):
        assert uu['index'] == oo['index']
        tokens = uu['tokens']
        yt_u = uu['y_true']
        yt_o = oo['y_true']
        assert yt_u == yt_o
        yp_u = uu['y_pred']
        yp_o = oo['y_pred']
        L = min(len(yp_u), len(yp_o), len(yt_u))
        for i in range(L):
            total += 1
            t = yt_u[i]
            by_true[t] += 1
            if yp_u[i] == yp_o[i]:
                agree += 1
            else:
                conf[(t, yp_u[i])][yp_o[i]] += 1
                if len(sample_diffs) < 30:  # store a few examples
                    sample_diffs.append({
                        'index': uu['index'],
                        'pos': i,
                        'token': tokens[i] if i < len(tokens) else None,
                        'y_true': t,
                        'y_pred_upgraded': yp_u[i],
                        'y_pred_original': yp_o[i],
                    })

    print(f'Total positions: {total}')
    print(f'Agree: {agree} ({agree/total:.4f})')
    print('\nCounts by true label:')
    for k, v in by_true.items():
        print(f'  {k}: {v}')

    print('\nTop 20 disagreement buckets (by (true, upgraded)->original counts):')
    flat = []
    for (t, up), m in conf.items():
        s = sum(m.values())
        flat.append((s, t, up, dict(m)))
    flat.sort(reverse=True)
    for s, t, up, d in flat[:20]:
        print(f'  true={t}, upgraded={up} -> {s} cases split as {d}')

    print('\nSample diffs:')
    for ex in sample_diffs:
        print(json.dumps(ex, ensure_ascii=False))

