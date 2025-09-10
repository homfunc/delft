#!/usr/bin/env python3
import argparse
import json
import os

from delft.sequenceLabelling import Sequence
from delft.sequenceLabelling.reader import load_data_and_labels_crf_file

def dump_predictions(input_path: str, output_path: str, model_name: str):
    # Load data
    x_all, y_all, f_all = load_data_and_labels_crf_file(input_path)

    # Load model by saved name
    seq = Sequence(model_name)
    seq.load()

    # Build generator identical to eval_single
    generator = seq.model.get_generator()
    test_generator = generator(
        x_all, y_all,
        batch_size=seq.model_config.batch_size,
        preprocessor=seq.p,
        char_embed_size=seq.model_config.char_embedding_size,
        max_sequence_length=seq.model_config.max_sequence_length,
        embeddings=seq.embeddings,
        shuffle=False,
        features=f_all,
        output_input_offsets=True,
        use_chain_crf=seq.model_config.use_chain_crf,
    )

    out_f = open(output_path, 'w', encoding='utf-8')

    idx = 0
    for i in range(len(test_generator)):
        data, label = test_generator[i]
        # For non-transformer architectures, data is a tuple of inputs ending with length_input.
        model_inputs = data

        # Predict
        y_pred_batch = seq.model.predict_on_batch(model_inputs)

        # If no CRF, take argmax; with CRF wrapper the decoded sequence is already indices
        if not seq.model_config.use_crf:
            import numpy as np
            y_pred_batch = np.argmax(y_pred_batch, -1)

        # Restore to original tokens using sequence lengths (last input is length_input)
        import numpy as np
        sequence_lengths = model_inputs[-1]
        sequence_lengths = np.reshape(sequence_lengths, (-1,))

        batch_size = len(sequence_lengths)
        for b in range(batch_size):
            true_indices = label[b]
            pred_indices = y_pred_batch[b]
            L = int(sequence_lengths[b])
            true_seq = seq.p.inverse_transform(true_indices[:L])
            pred_seq = seq.p.inverse_transform(pred_indices[:L])
            obj = {
                'index': idx,
                'tokens': x_all[idx],
                'y_true': true_seq,
                'y_pred': pred_seq,
            }
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            idx += 1

    out_f.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to CRF-format data file (same as used for eval)')
    ap.add_argument('--output', required=True, help='Path to output JSONL with per-sequence predictions')
    ap.add_argument('--model-name', default='grobid-date-BidLSTM_CRF', help='Saved model name directory under data/models/sequenceLabelling')
    args = ap.parse_args()

    dump_predictions(args.input, args.output, args.model_name)

