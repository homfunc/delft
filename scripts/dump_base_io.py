#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import numpy as np
import keras
import json

from delft.sequenceLabelling import Sequence
from delft.sequenceLabelling.reader import load_data_and_labels_crf_file


def to_numpy(x):
    try:
        return keras.ops.convert_to_numpy(x)
    except Exception:
        try:
            return np.array(x)
        except Exception:
            return None


def deterministic_subset(x_all: List[List[str]], y_all: List[List[str]], f_all, k: int) -> Tuple[List[List[str]], List[List[str]], List]:
    rng = np.random.default_rng(seed=42)
    sel = rng.choice(len(x_all), k, replace=False, shuffle=False)
    x_sel = [x_all[i] for i in sel]
    y_sel = [y_all[i] for i in sel] if y_all is not None else None
    f_sel = [f_all[i] for i in sel] if f_all is not None else None
    return x_sel, y_sel, f_sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--input', required=True, help='Path to CRF-format data file')
    ap.add_argument('--out', required=True, help='Path to output .npz file')
    ap.add_argument('--limit', type=int, default=8)
    ap.add_argument('--deterministic', action='store_true', help='Use a deterministic subset selection of size --limit')
    ap.add_argument('--fail-if-keras-exists', action='store_true', help='Exit with error if model.keras exists in the model directory')
    args = ap.parse_args()

    seq = Sequence(args.model_name)
    model_dir = os.path.join('data/models/sequenceLabelling', args.model_name)
    keras_path = os.path.join(model_dir, 'model.keras')
    if args.fail_if_keras_exists and os.path.exists(keras_path):
        raise SystemExit(f"Refusing to load from model.keras; please remove {keras_path} first")

    # Load model (may create model.keras after load; the caller can remove it if needed)
    seq.load()

    # Base model before CRF wrapper
    wrapper = seq.model
    base = getattr(wrapper, 'base_model', None)
    if base is None:
        # Fallback: use wrapper directly if no CRF wrapper
        base = getattr(wrapper, 'model', wrapper)

    # Resolve layer handles by name (architecture-dependent)
    def get_layer_output(layer_name: str):
        try:
            return base.get_layer(layer_name).output
        except Exception as e:
            raise RuntimeError(f"Could not resolve layer '{layer_name}' in base model: {e}")

    # Typical names from summary: time_distributed (char Embedding TD), char_repr (final char per token), concatenate, bidirectional_1, dense
    E = get_layer_output('time_distributed')   # char Embedding TimeDistributed output (B, T, max_char, emb_dim)
    try:
        A = get_layer_output('char_repr')  # final per-token char vector
    except RuntimeError:
        # Fallback schemes for older names
        try:
            A = get_layer_output('time_distributed_2')
        except RuntimeError:
            A = get_layer_output('time_distributed_1')
    B = get_layer_output('concatenate')         # concat word+chars
    # Bidirectional layer may also shift name; try bidirectional_1, fallback to bidirectional
    try:
        C = get_layer_output('bidirectional_1')
    except RuntimeError:
        C = get_layer_output('bidirectional')
    D = get_layer_output('dense')              # pre-CRF Dense tanh

    probe = keras.Model(inputs=base.inputs, outputs=[E, A, B, C, D], name='base_probe')

    # Prepare batch inputs like eval path
    x_all, y_all, f_all = load_data_and_labels_crf_file(args.input)
    if args.deterministic:
        x_all, y_all, f_all = deterministic_subset(x_all, y_all, f_all, args.limit)
    else:
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
    model_inputs = data  # [word_input, char_input, length_input]

    # Save the actual tokens and labels used for this batch (JSON next to NPZ)
    tokens_json_path = args.out + '.tokens.json'
    try:
        os.makedirs(os.path.dirname(tokens_json_path), exist_ok=True)
        with open(tokens_json_path, 'w', encoding='utf-8') as jf:
            json.dump({
                'tokens': x_all,
                'labels': y_all,
            }, jf, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to write tokens JSON {tokens_json_path}: {e}")

    e_t, a_t, b_t, c_t, d_t = probe(model_inputs, training=False)
    E_np = to_numpy(e_t)
    A_np = to_numpy(a_t)
    B_np = to_numpy(b_t)
    C_np = to_numpy(c_t)
    D_np = to_numpy(d_t)
    if any(x is None for x in (E_np, A_np, B_np, C_np, D_np)):
        raise RuntimeError('Failed to convert stage tensors to numpy')

    # lengths and time mask
    lengths = to_numpy(model_inputs[2]).reshape((-1,))
    Bsz, T = A_np.shape[0], A_np.shape[1]
    t_range = np.arange(T)[None, :]
    mask = (t_range < lengths[:, None])

    # Dump char input ids and per-token char masks/lengths
    char_ids = to_numpy(model_inputs[1])  # shape (B, T, max_char)
    if char_ids is None:
        raise RuntimeError('Failed to extract char_ids from model inputs')
    char_mask = (char_ids != 0).astype(np.uint8)  # per-char mask within token
    char_lengths = char_mask.sum(axis=-1).astype(np.int32)  # per-token char length

    # Locate char embedding weights and time_distributed_1 weights
    def get_path_or_name(w):
        return getattr(w, 'path', getattr(w, 'name', ''))

    char_emb = None
    td1_weights = {}
    for w in base.weights:
        pname = get_path_or_name(w)
        if 'char_embeddings/embeddings' in pname and char_emb is None:
            arr = to_numpy(w)
            if arr is not None:
                char_emb = arr
        if '/time_distributed_1/' in pname:
            arr = to_numpy(w)
            if arr is not None:
                key = 'w__' + pname.replace('/', '__').replace(':', '_')
                td1_weights[key] = arr

    # Deep-extract TD1 Bidirectional(LSTM) cell weights if possible
    try:
        td1_layer = base.get_layer('time_distributed_1')
        inner = getattr(td1_layer, 'layer', None)
        bidir = inner if inner is not None else td1_layer
        # Try to access forward/backward LSTM layers
        fwd = getattr(bidir, 'forward_layer', None)
        bwd = getattr(bidir, 'backward_layer', None)
        def dump_lstm(prefix, lstm):
            if lstm is None:
                return
            # Preferred attribute access
            k = getattr(lstm, 'kernel', None)
            rk = getattr(lstm, 'recurrent_kernel', None)
            b = getattr(lstm, 'bias', None)
            if k is not None:
                arr = to_numpy(k)
                if arr is not None:
                    td1_weights[f'{prefix}_kernel'] = arr
            if rk is not None:
                arr = to_numpy(rk)
                if arr is not None:
                    td1_weights[f'{prefix}_recurrent_kernel'] = arr
            if b is not None:
                arr = to_numpy(b)
                if arr is not None:
                    td1_weights[f'{prefix}_bias'] = arr
            # Fallback to get_weights ordering if attrs not present
            if (k is None or rk is None or b is None):
                try:
                    gw = lstm.get_weights()
                    if len(gw) >= 3:
                        if f'{prefix}_kernel' not in td1_weights:
                            td1_weights[f'{prefix}_kernel'] = gw[0]
                        if f'{prefix}_recurrent_kernel' not in td1_weights:
                            td1_weights[f'{prefix}_recurrent_kernel'] = gw[1]
                        if f'{prefix}_bias' not in td1_weights:
                            td1_weights[f'{prefix}_bias'] = gw[2]
                except Exception:
                    pass
        dump_lstm('td1_fwd', fwd)
        dump_lstm('td1_bwd', bwd)
    except Exception:
        pass

    save_kwargs = dict(
        E=E_np, A=A_np, B=B_np, C=C_np, D=D_np,
        lengths=lengths,
        mask=mask.astype(np.uint8),
        char_ids=char_ids,
        char_mask=char_mask,
        char_lengths=char_lengths,
    )
    if char_emb is not None:
        save_kwargs['char_emb'] = char_emb
    # Merge td1 weights into kwargs
    for k, v in td1_weights.items():
        save_kwargs[k] = v

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **save_kwargs)
    print(f"Saved base IO to {args.out}:")
    print('  A:', A_np.shape, 'B:', B_np.shape, 'C:', C_np.shape, 'D:', D_np.shape)
    print('  lengths:', lengths.tolist())
    print('  char_ids:', char_ids.shape, 'char_emb:', None if char_emb is None else char_emb.shape, f'td1_weights: {len(td1_weights)} items')
