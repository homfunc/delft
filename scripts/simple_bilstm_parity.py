#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import keras
from keras.layers import Input, Embedding, LSTM, Bidirectional, TimeDistributed, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

# Reproducibility
np.random.seed(1234)


def gen_data(out_path: str,
             n_train: int = 300,
             n_test: int = 100,
             max_seq_len: int = 20,
             max_char_len: int = 15,
             n_tags: int = 5,
             char_vocab_size: int = 60):
    """Generate deterministic synthetic sequence labeling data.
    - char ids in [1..char_vocab_size-1], 0 is padding
    - labels in [0..n_tags-1]
    """
    rng = np.random.RandomState(1337)

    def make_split(n):
        lengths = rng.randint(2, max_seq_len + 1, size=(n,), dtype=np.int32)
        char_ids = np.zeros((n, max_seq_len, max_char_len), dtype=np.int32)
        labels = np.zeros((n, max_seq_len), dtype=np.int32)
        for i in range(n):
            L = int(lengths[i])
            for t in range(L):
                clen = rng.randint(1, max_char_len + 1)
                char_ids[i, t, :clen] = rng.randint(1, char_vocab_size, size=(clen,), dtype=np.int32)
                labels[i, t] = rng.randint(0, n_tags)
        return char_ids, labels, lengths

    Xtr_char, ytr, Ltr = make_split(n_train)
    Xte_char, yte, Lte = make_split(n_test)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
             train_char=Xtr_char,
             train_labels=ytr,
             train_lengths=Ltr,
             test_char=Xte_char,
             test_labels=yte,
             test_lengths=Lte,
             n_tags=n_tags,
             max_seq_len=max_seq_len,
             max_char_len=max_char_len,
             char_vocab_size=char_vocab_size)
    print(f"Saved dataset to {out_path}")


def build_model(n_tags: int,
                char_vocab_size: int,
                char_emb_dim: int = 16,
                char_lstm_units: int = 16,
                word_lstm_units: int = 32,
                max_char_len: int = 15,
                dropout: float = 0.1) -> Model:
    # Inputs
    char_input = Input(shape=(None, max_char_len), dtype='int32', name='char_input')

    # TimeDistributed Embedding with mask_zero=True
    char_emb = TimeDistributed(
        Embedding(input_dim=char_vocab_size,
                  output_dim=char_emb_dim,
                  mask_zero=True,
                  name='char_embeddings'),
        name='td_char_emb')(char_input)

    # Deterministic char representation with pinned LSTM params and explicit gather
    from keras import ops as K

    # Compute per-token char lengths from char_input (count non-zero ids)
    lengths = K.sum(K.cast(K.not_equal(char_input, 0), 'int32'), axis=-1)  # (B,T)

    # Forward and backward LSTMs over char axis with full sequences
    f_seq = TimeDistributed(LSTM(
        char_lstm_units,
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        unit_forget_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        implementation=1,
        recurrent_dropout=0.0,
    ), name='char_lstm_fwd')(char_emb)
    b_seq = TimeDistributed(LSTM(
        char_lstm_units,
        return_sequences=True,
        go_backwards=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        unit_forget_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        implementation=1,
        recurrent_dropout=0.0,
    ), name='char_lstm_bwd')(char_emb)

    # Gather last valid forward step and first backward step using one-hot select
    idx_f = K.maximum(lengths - 1, 0)  # (B,T)
    idx_b = K.zeros_like(lengths)

    C = K.shape(f_seq)[2]
    oh_f = K.one_hot(idx_f, C)  # (B,T,C)
    oh_f = K.cast(oh_f, f_seq.dtype)
    f_last = K.sum(f_seq * K.expand_dims(oh_f, axis=-1), axis=2)  # (B,T,H)

    oh_b = K.one_hot(idx_b, C)
    oh_b = K.cast(oh_b, b_seq.dtype)
    b_first = K.sum(b_seq * K.expand_dims(oh_b, axis=-1), axis=2)  # (B,T,H)

    char_td = keras.layers.Concatenate(name='char_repr')([f_last, b_first])

    x = Dropout(dropout, name='drop_char')(char_td)
    x = Bidirectional(LSTM(word_lstm_units, return_sequences=True), name='word_bilstm')(x)
    x = Dropout(dropout, name='drop_word')(x)
    logits = Dense(n_tags, activation='softmax', name='classifier')(x)

    model = Model(inputs=[char_input], outputs=[logits], name='simple_bilstm')
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate(data_path: str, weights_path: str, batch_size: int = 32):
    Z = np.load(data_path)
    Xc = Z['test_char']
    y = Z['test_labels']
    L = Z['test_lengths']
    n_tags = int(Z['n_tags'])
    max_char_len = int(Z['max_char_len'])
    char_vocab_size = int(Z['char_vocab_size'])

    model = build_model(n_tags=n_tags,
                        char_vocab_size=char_vocab_size,
                        max_char_len=max_char_len)
    try:
        model.load_weights(weights_path)
    except Exception as e:
        # As a fallback, attempt by-name assignment using legacy H5 loader if available in project
        try:
            from delft.utilities.weights import load_weights_by_name_from_h5
            load_weights_by_name_from_h5(model, weights_path, verbose=True, strict=False)
        except Exception:
            raise e

    T = Xc.shape[1]
    tmask = (np.arange(T)[None, :] < L[:, None]).astype('float32')

    loss, acc = model.evaluate([Xc], y, sample_weight=tmask, batch_size=batch_size, verbose=0)

    print(json.dumps({'loss': float(loss), 'accuracy': float(acc), 'micro_f1': float(acc)}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap_g = sub.add_parser('gen-data')
    ap_g.add_argument('--out', required=True)

    ap_e = sub.add_parser('eval')
    ap_e.add_argument('--data', required=True)
    ap_e.add_argument('--weights', required=True)
    ap_e.add_argument('--batch-size', type=int, default=32)

    args = ap.parse_args()

    if args.cmd == 'gen-data':
        gen_data(args.out)
    elif args.cmd == 'eval':
        evaluate(args.data, args.weights, batch_size=args.batch_size)

