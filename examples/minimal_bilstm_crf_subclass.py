#!/usr/bin/env python3
import os
import numpy as np
import keras
from keras import layers

from keras_crf.layers import CRF
from keras_crf.losses import nll_loss as crf_nll_loss


@keras.saving.register_keras_serializable(package="DeLFTExamples", name="MinimalBiLSTMCRF")
class MinimalBiLSTMCRF(keras.Model):
    def __init__(self, num_tags:int, vocab_size:int, embedding_dim:int=64, lstm_units:int=64, **kwargs):
        name = kwargs.pop('name', "minimal_bilstm_crf")
        super().__init__(name=name, **kwargs)
        self.num_tags = int(num_tags)
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.lstm_units = int(lstm_units)
        self.embed = layers.Embedding(self.vocab_size + 1, self.embedding_dim, mask_zero=True, name="embed")
        self.bilstm = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True), name="word_bilstm")
        self.proj = layers.Dense(self.num_tags, name="tanh_proj")
        self.crf = CRF(self.num_tags, name="crf")
        self._cache = None

    def call(self, tokens):
        x = self.embed(tokens)
        x = self.bilstm(x)
        x = self.proj(x)
        decoded, potentials, lens, trans = self.crf(x)
        self._cache = (potentials, lens)
        return decoded

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        pot, lens = self._cache if self._cache is not None else (None, None)
        if pot is None:
            _ = self(x)
            pot, lens = self._cache
        loss = crf_nll_loss(pot, lens, self.crf.trans, y, sample_weight=sample_weight, reduction="mean")
        return loss

    def compute_output_shape(self, input_shape):
        try:
            b, t = input_shape
        except Exception:
            b, t = None, None
        return (b, t)

    def compute_output_spec(self, inputs, batch_size=None, dtype=None):
        if isinstance(inputs, (list, tuple)):
            inp = inputs[0]
        else:
            inp = inputs
        shp = getattr(inp, "shape", None)
        if shp is not None:
            b, t = shp[0], shp[1]
        else:
            b, t = None, None
        # Return a KerasTensor describing the output without tracing CRF decode
        return keras.KerasTensor(shape=(b, t), dtype="int32")

    def get_config(self):
        base = super().get_config()
        base.update({
            "num_tags": self.num_tags,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
        })
        return base

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

    def get_build_config(self):
        return {"input_specs": [{"name": "tokens", "dtype": "int32", "shape": (None,)}]}

    def build_from_config(self, cfg):
        # Purely symbolic forward to build all variables (CRF safe under tracing)
        tok = keras.Input(shape=(None,), dtype="int32", name="tokens")
        _ = self(tok)
        self.built = True


def make_varlen_dataset(n=1000, max_len=30, vocab_size=200, num_tags=6, seed=0):
    rng = np.random.default_rng(seed)
    lens = rng.integers(1, max_len+1, size=n)
    X = np.zeros((n, max_len), dtype=np.int32)
    Y = np.zeros((n, max_len), dtype=np.int32)
    for i, L in enumerate(lens):
        X[i, :L] = rng.integers(1, vocab_size+1, size=L)
        Y[i, :L] = rng.integers(0, num_tags, size=L)
    return X, Y


def main():
    # Synthetic tiny dataset
    X_train, Y_train = make_varlen_dataset(800, 40, 300, 6, seed=1)
    X_val, Y_val = make_varlen_dataset(200, 40, 300, 6, seed=2)
    X_test, Y_test = make_varlen_dataset(200, 40, 300, 6, seed=3)

    model = MinimalBiLSTMCRF(num_tags=6, vocab_size=300, embedding_dim=64, lstm_units=64)
    model.compile(optimizer=keras.optimizers.Adam(1e-3))

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=1, batch_size=64, verbose=2)

    # Inference before save
    pred = model.predict(X_test, batch_size=64, verbose=0)

    # Save and reload
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)
    keras_path = os.path.join(out_dir, 'minimal_bilstm_crf_subclass.keras')
    model.save(keras_path)

    loaded = keras.models.load_model(keras_path)
    pred2 = loaded.predict(X_test, batch_size=64, verbose=0)

    assert pred2.shape == pred.shape, f"shape mismatch: {pred2.shape} vs {pred.shape}"
    if not np.array_equal(pred2, pred):
        mismatch = np.mean(pred2 != pred)
        raise SystemExit(f"Reloaded predictions differ: mismatch rate {mismatch:.4f}")

    print("Minimal subclass roundtrip OK; predictions identical.")


if __name__ == "__main__":
    main()

