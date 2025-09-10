#!/usr/bin/env python3
"""
Minimal DeLFT-style CRF wrapper example (backend-agnostic, Keras 3).

This demonstrates how to build a multi-input encoder (similar in spirit to DeLFT)
and keep the CRF loss fully inside the model graph using keras-crf.losses utilities
(no eager mode, no fixed sequence length).

Inputs
- tokens: [B, T] integer token ids (for masking and word embedding)
- char_input: [B, T, max_char] integer char ids (dummy in this demo)
- length_input: [B, 1] true sequence lengths
- labels: [B, T] gold tag ids

Outputs
- decoded_output: [B, T] decoded tag ids (metrics only)
- crf_loss_value: [B] per-sample loss vector (mean used as training loss)

How to run (from the keras-crf repo root):
  python examples/delft_crf_wrapper_example.py --epochs 3 --batch-size 64

This script uses keras_crf.losses (nll via crf_log_likelihood) instead of
reinventing any loss logic in the wrapper.
"""

import argparse
import os
import sys

import numpy as np
import keras
from keras import layers, ops as K

# Ensure local examples utils are importable when running as a script
EXAMPLES_ROOT = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(EXAMPLES_ROOT)
sys.path.insert(0, REPO_ROOT)

from keras_crf.layers import CRF
from keras_crf.losses import crf_log_likelihood
from examples.utils.data import make_varlen_dataset
from examples.utils.metrics import MaskedTokenAccuracy


def build_delft_style_encoder(num_tags: int, vocab_size: int, max_char: int = 30,
                              word_dim: int = 128, char_dim: int = 16, word_lstm_units: int = 64,
                              char_lstm_units: int = 16):
    # Inputs (names loosely mirror DeLFT)
    tokens_in = keras.Input(shape=(None,), dtype="int32", name="tokens")
    char_input = keras.Input(shape=(None, max_char), dtype="int32", name="char_input")
    length_input = keras.Input(shape=(1,), dtype="int32", name="length_input")

    # Word channel (masking on 0)
    w = layers.Embedding(vocab_size + 1, word_dim, mask_zero=False, name="word_embed")(tokens_in)

    # Char channel (TimeDistributed over tokens)
    # Note: This demo ignores real char ids and uses a simple embedding -> BiLSTM pooling
    c = layers.TimeDistributed(layers.Embedding(input_dim=128, output_dim=char_dim, mask_zero=False), name="char_embed_td")(char_input)
    c = layers.TimeDistributed(layers.Bidirectional(layers.LSTM(char_lstm_units, return_sequences=False)), name="char_bilstm_td")(c)

    # Concatenate channels
    x = layers.Concatenate(name="concat_word_char")([w, c])
    x = layers.Dropout(0.2)(x)

    # Word-level BiLSTM
    x = layers.Bidirectional(layers.LSTM(word_lstm_units, return_sequences=True), name="bilstm")(x)
    x = layers.Dropout(0.2)(x)

    # Small projection before CRF
    feats = layers.Dense(word_lstm_units, activation="tanh", name="proj")(x)


    # CRF layer
    crf = CRF(num_tags)
    decoded, potentials, lengths, trans = crf(feats)

    # In-graph CRF negative log-likelihood
    labels = keras.Input(shape=(None,), dtype="int32", name="labels")
    # Flatten provided length_input to shape [B]
    lengths_flat = keras.layers.Lambda(lambda x: K.squeeze(x, axis=-1), name="lengths_flat")(length_input)

    class CRF_NLL_Layer(keras.layers.Layer):
        def call(self, inputs):
            pot, y_true, ln, tr = inputs
            ll = crf_log_likelihood(pot, y_true, ln, tr)
            return -ll  # [B]

        def compute_output_shape(self, input_shape):
            # Return a per-sample loss vector: shape (batch,)
            return (None,)

    # Use external lengths (from input) to compute NLL to avoid relying on mask propagation
    nll_vec = CRF_NLL_Layer(name="nll_out")([potentials, labels, lengths_flat, trans])

    # Name outputs to align with compile dict
    decoded_named = keras.layers.Lambda(lambda z: z, name="decoded_output")(decoded)
    loss_named = keras.layers.Lambda(lambda z: z, name="crf_loss_value")(nll_vec)

    model = keras.Model(inputs=[tokens_in, char_input, length_input, labels],
                        outputs=[decoded_named, loss_named],
                        name="delft_crf_wrapper_demo")

    # Zero-loss for decoded head; training signal is mean of per-sample NLL
    def zero_loss(y_true, y_pred):
        return K.mean(K.zeros_like(y_pred[..., :1]))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"decoded_output": zero_loss, "crf_loss_value": lambda y_true, y_pred: K.mean(y_pred)},
        metrics={"decoded_output": [MaskedTokenAccuracy()]},
    )

    # Inference model returning decoded tags only
    infer = keras.Model([tokens_in, char_input, length_input], decoded_named, name="delft_crf_wrapper_infer")
    return model, infer


def parse_args():
    p = argparse.ArgumentParser(description="Minimal DeLFT-style CRF wrapper example using keras_crf.losses")
    p.add_argument("--synthetic-samples", type=int, default=2000)
    p.add_argument("--synthetic-max-len", type=int, default=40)
    p.add_argument("--synthetic-vocab", type=int, default=500)
    p.add_argument("--synthetic-tags", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-char", type=int, default=30)
    return p.parse_args()


def main():
    a = parse_args()

    # Synthetic variable-length dataset
    X_train, Y_train, _ = make_varlen_dataset(a.synthetic_samples, a.synthetic_max_len,
                                             a.synthetic_vocab, a.synthetic_tags, seed=1)
    X_val, Y_val, _ = make_varlen_dataset(max(a.synthetic_samples // 5, 1), a.synthetic_max_len,
                                         a.synthetic_vocab, a.synthetic_tags, seed=2)

    num_tags = a.synthetic_tags
    vocab_size = a.synthetic_vocab

    # Derive sequence lengths per sample (mask: token != 0)
    L_train = (X_train != 0).sum(axis=1, keepdims=True).astype("int32")
    L_val = (X_val != 0).sum(axis=1, keepdims=True).astype("int32")

    # Dummy char inputs (zeros) for demonstration; shape [B, T, max_char]
    C_train = np.zeros((X_train.shape[0], X_train.shape[1], a.max_char), dtype="int32")
    C_val = np.zeros((X_val.shape[0], X_val.shape[1], a.max_char), dtype="int32")

    model, infer = build_delft_style_encoder(num_tags=num_tags, vocab_size=vocab_size,
                                             max_char=a.max_char)

    # Train: decoded_output has zero loss; CRF loss comes from crf_loss_value
    # Provide dummy y for decoded head (just reuse Y arrays); sample weights mask the metric head only
    sw_decoded_train = (X_train != 0).astype("float32")
    sw_decoded_val = (X_val != 0).astype("float32")

    history = model.fit(
        x={"tokens": X_train, "char_input": C_train, "length_input": L_train, "labels": Y_train},
        y={"decoded_output": Y_train, "crf_loss_value": np.zeros((X_train.shape[0],), dtype="float32")},
        validation_data=(
            {"tokens": X_val, "char_input": C_val, "length_input": L_val, "labels": Y_val},
            {"decoded_output": Y_val, "crf_loss_value": np.zeros((X_val.shape[0],), dtype="float32")},
        ),
        epochs=a.epochs,
        batch_size=a.batch_size,
        verbose=2,
    )

    # Quick sanity-check inference
    decoded_val = infer.predict([X_val, C_val, L_val], batch_size=a.batch_size, verbose=0)
    acc = (decoded_val[X_val != 0] == Y_val[X_val != 0]).mean()
    print(f"Masked token accuracy (val): {acc:.4f}")


if __name__ == "__main__":
    main()

