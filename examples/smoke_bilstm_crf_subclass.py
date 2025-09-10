#!/usr/bin/env python3
"""
Smoke test for BidLSTM_CRF_Subclass

- Builds a tiny dataset and preprocessor
- Instantiates the BidLSTM_CRF_Subclass via the factory
- Runs a short fit to create variables
- Saves to native Keras format (.keras) and reloads

This script avoids external embeddings by generating small random word vectors.
"""
import os
import numpy as np
import keras

from delft.sequenceLabelling.preprocess import prepare_preprocessor
from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.models import get_model


def main():
    # Tiny toy dataset
    X_train = [["John","lives","in","Paris"],
               ["Mary","works","at","Acme"],
               ["I","love","New","York"]]
    y_train = [["B-PER","O","O","B-LOC"],
               ["B-PER","O","O","B-ORG"],
               ["O","O","B-LOC","I-LOC"]]

    X_valid = [["Bob","is","from","London"]]
    y_valid = [["B-PER","O","O","B-LOC"]]

    # Model configuration (using precomputed word embeddings of size 32)
    cfg = ModelConfig(
        model_name="smoke_bilstm_crf_subclass_manual",
        architecture="BidLSTM_CRF_Subclass",
        embeddings_name="dummy-local",
        word_embedding_size=32,
        char_emb_size=16,
        char_lstm_units=8,
        word_lstm_units=16,
        max_char_length=10,
        dropout=0.2,
        recurrent_dropout=0.0,
        batch_size=2,
    )

    # Prepare preprocessor and update derived sizes
    p = prepare_preprocessor(X_train + X_valid, y_train + y_valid, model_config=cfg)
    cfg.char_vocab_size = len(p.vocab_char)

    # Get model wrapper and underlying Keras model
    mdl_wrapper = get_model(cfg, p, ntags=len(p.vocab_tag), load_pretrained_weights=False)
    mdl = mdl_wrapper.model

    # Build inputs for training: char IDs, lengths, random word vectors aligned to max T
    (batches_train, y_idx_train) = p.transform(X_train, y_train, label_indices=True)
    char_train = np.asarray(batches_train[0], dtype=np.int32)
    len_train = batches_train[1]
    T = char_train.shape[1]

    class _DummyEmb:
        def __init__(self, d=32): self.d = d
        def get(self, w):
            rng = np.random.default_rng(abs(hash(w)) % (2**32))
            return rng.uniform(-0.5, 0.5, size=(self.d,)).astype('float32')

    demb = _DummyEmb(32)
    Xw_train = np.stack([
        np.stack([demb.get(w) for w in x] + [np.zeros(32, dtype='float32')] * (T - len(x)))
        for x in X_train
    ]).astype('float32')
    y_train_np = np.asarray(y_idx_train, dtype=np.int32)

    # Compile and fit briefly
    mdl.compile(optimizer=keras.optimizers.Adam(1e-3))
    mdl.fit([Xw_train, char_train, len_train], y_train_np, epochs=1, batch_size=2, verbose=1)

    # Save in native Keras format and patch CRF variable names
    out_dir = os.path.join('data/models/sequenceLabelling', cfg.model_name)
    os.makedirs(out_dir, exist_ok=True)
    keras_path = os.path.join(out_dir, 'model.keras')
    mdl.save(keras_path)
    print('Saved to', keras_path)

    # Load back to validate
    loaded = keras.models.load_model(keras_path)
    print('Loaded native model:', type(loaded))
    print('OK')


if __name__ == '__main__':
    main()

