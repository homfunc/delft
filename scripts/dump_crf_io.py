#!/usr/bin/env python3
import argparse
import os
import numpy as np
import keras
from delft.sequenceLabelling import Sequence
from delft.sequenceLabelling.reader import load_data_and_labels_crf_file
from typing import List, Tuple

def to_numpy(x):
    try:
        return keras.ops.convert_to_numpy(x)
    except Exception:
        try:
            return np.array(x)
        except Exception:
            return None

def deterministic_subset(x_all: List[List[str]], y_all: List[List[str]], f_all, k: int) -> Tuple[List[List[str]], List[List[str]], List]:
    """
    Deterministically select k samples by using random choice of k with fixed seed of 42.
    """
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
    args = ap.parse_args()

    seq = Sequence(args.model_name)
    seq.load()

    inner = getattr(seq.model, 'model', seq.model)

    # Prepare small batch identical to eval path
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
    model_inputs = data

    # Ask the model to return CRF internals
    internals, decoded = inner(model_inputs, training=False, return_crf_internal=True)
    potentials, lengths, trans = internals

    pot_np = to_numpy(potentials)
    len_np = to_numpy(lengths)
    trans_np = to_numpy(trans)
    dec_np = to_numpy(decoded)

    if pot_np is None or len_np is None or trans_np is None or dec_np is None:
        raise RuntimeError("Failed to convert one or more tensors to numpy arrays")

    # Derive mask/first/last indices from lengths and batch time dimension (RIGHT-PADDING)
    B, T, N = pot_np.shape
    # valid tokens occupy indices [0 .. L-1]
    first_idx = np.zeros((B,), dtype=len_np.dtype)
    last_idx = len_np - 1
    # boolean mask [B,T]: True for valid tokens
    t_range = np.arange(T)[None, :]
    mask = (t_range < len_np[:, None])

    # Persist arrays for cross-repo comparison
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out,
             potentials=pot_np,
             lengths=len_np,
             transitions=trans_np,
             decoded=dec_np,
             first_idx=first_idx,
             last_idx=last_idx,
             mask=mask.astype(np.uint8))
    print(f"Saved CRF IO to {args.out}: pot={pot_np.shape}, len={len_np.shape}, trans={trans_np.shape}, dec={dec_np.shape}")
    # Print explicit sequence lengths for verification
    print("sequence_lengths:", len_np.tolist())

