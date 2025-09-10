#!/usr/bin/env python3
import argparse
import json
import numpy as np

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
    ap.add_argument('--limit', type=int, default=16)
    args = ap.parse_args()

    seq = Sequence(args.model_name)
    seq.load()

    # Use the inner Keras model (architecture wrapper forwards attributes but is not callable)
    inner_model = getattr(seq.model, 'model', seq.model)

    # Introspect CRF layer weights
    crf = getattr(inner_model, 'crf', None)
    weights_info = []
    if crf is not None:
        for w in crf.weights:
            try:
                val = w.numpy()
            except Exception:
                try:
                    val = w.value().numpy()
                except Exception:
                    val = None
            info = {
                'name': w.name,
                'trainable': bool(getattr(w, 'trainable', True)),
                'summary': tensor_summary(val) if val is not None else None,
            }
            weights_info.append(info)

    # Build a small batch and dump internals
    x_all, y_all, f_all = load_data_and_labels_crf_file(args.input)
    x_all = x_all[:args.limit]
    y_all = y_all[:args.limit]
    f_all = f_all[:args.limit] if f_all is not None else None

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
    model_inputs = data  # non-transformer path

    internals = None
    try:
        internals, decoded = inner_model(model_inputs, training=False, return_crf_internal=True)
        potentials, lengths, trans = internals
        internals = {
            'potentials': tensor_summary(potentials),
            'lengths': tensor_summary(lengths),
            'transitions': tensor_summary(trans),
        }
    except Exception as e:
        internals = {'error': str(e)}
        decoded = inner_model.predict_on_batch(model_inputs)

    out = {
        'weights': weights_info,
        'internals': internals,
        'decoded_shape': list(np.asarray(decoded).shape),
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

