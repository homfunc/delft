#!/usr/bin/env python3
import argparse
import json
import numpy as np
import re

from delft.sequenceLabelling import Sequence
from delft.sequenceLabelling.reader import load_data_and_labels_crf_file

def tensor_summary(arr):
    a = np.asarray(arr)
    return {
        'shape': list(a.shape),
        'dtype': str(a.dtype),
        'min': float(np.min(a)) if a.size else 0.0,
        'max': float(np.max(a)) if a.size else 0.0,
        'mean': float(np.mean(a)) if a.size else 0.0,
        'std': float(np.std(a)) if a.size else 0.0,
        'l2': float(np.linalg.norm(a)) if a.size else 0.0,
    }

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', default='grobid-date-BidLSTM_CRF')
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=32)
    args = ap.parse_args()

    # Load data
    x_all, y_all, f_all = load_data_and_labels_crf_file(args.input)
    x_all = x_all[:args.limit]
    y_all = y_all[:args.limit]
    f_all = f_all[:args.limit] if f_all is not None else None

    # Load model and get inner/base models
    seq = Sequence(args.model_name)
    seq.load()
    inner_model = getattr(seq.model, 'model', seq.model)
    base_model = getattr(inner_model, 'base_model', None)
    if base_model is None:
        raise RuntimeError('No base_model found')

    # Build generator
    gen = seq.model.get_generator()
    test_gen = gen(
        x_all, y_all,
        batch_size=min(args.limit, seq.model_config.batch_size),
        preprocessor=seq.p,
        char_embed_size=seq.model_config.char_embedding_size,
        max_sequence_length=seq.model_config.max_sequence_length,
        embeddings=seq.embeddings,
        shuffle=False,
        features=f_all,
        output_input_offsets=True,
        use_chain_crf=seq.model_config.use_chain_crf,
    )

    data, label = test_gen[0]
    model_inputs = data

    # Run base model to get pre-CRF features
    feats = base_model.predict_on_batch(model_inputs)

    out = {
        'feats': tensor_summary(feats),
    }

    # Optional: extract a few positions where tokens look numeric (potential month/day confusions)
    numeric_positions = []
    num_pat = re.compile(r'^[0-9]{1,4}$')
    # length_input is last
    import numpy as np
    seq_lengths = np.reshape(model_inputs[-1], (-1,))
    B, T, F = feats.shape
    for b in range(min(B, len(x_all))):
        L = int(seq_lengths[b]) if b < len(seq_lengths) else T
        tokens = x_all[b]
        for t in range(min(L, len(tokens))):
            tok = tokens[t]
            if num_pat.match(tok):
                # record feature vector norm for a small sample
                v = feats[b, t]
                numeric_positions.append({
                    'b': b,
                    't': t,
                    'token': tok,
                    'feat_l2': float(np.linalg.norm(v)),
                    'feat_min': float(np.min(v)),
                    'feat_max': float(np.max(v)),
                    'feat_mean': float(np.mean(v)),
                    'feat_std': float(np.std(v)),
                })
                if len(numeric_positions) >= 40:
                    break
        if len(numeric_positions) >= 40:
            break

    out['numeric_positions'] = numeric_positions

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

