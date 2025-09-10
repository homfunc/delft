#!/usr/bin/env python3
import os
import numpy as np
import keras
from keras import layers, ops as K

from keras_crf.layers import CRF
from keras_crf.losses import nll_loss as crf_nll_loss

# Utility: small synthetic dataset of variable lengths

def make_varlen_dataset(n=200, max_len=20, vocab_size=200, num_tags=6, seed=0):
    rng = np.random.default_rng(seed)
    lens = rng.integers(1, max_len + 1, size=n)
    X = np.zeros((n, max_len), dtype=np.int32)
    Y = np.zeros((n, max_len), dtype=np.int32)
    for i, L in enumerate(lens):
        X[i, :L] = rng.integers(1, vocab_size + 1, size=L)
        Y[i, :L] = rng.integers(0, num_tags, size=L)
    return X, Y


def save_and_reload(model, x, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'model.keras')
    y_before = model(x)
    model.save(path)
    loaded = keras.models.load_model(path)
    y_after = loaded(x)
    np.testing.assert_allclose(K.convert_to_numpy(y_before), K.convert_to_numpy(y_after), rtol=1e-6, atol=1e-6)
    return path


# Step 1: Minimal subclass (Embedding -> BiLSTM -> CRF)
@keras.saving.register_keras_serializable(package="Probe", name="Step1_MinimalBiLSTMCRF")
class Step1_Minimal(keras.Model):
    def __init__(self, num_tags: int, vocab_size: int, embedding_dim: int = 64, lstm_units: int = 64, **kwargs):
        name = kwargs.pop('name', "step1_minimal")
        super().__init__(name=name, **kwargs)
        self.num_tags = int(num_tags)
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.lstm_units = int(lstm_units)
        self.embed = layers.Embedding(self.vocab_size + 1, self.embedding_dim, mask_zero=True)
        self.bilstm = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))
        self.crf = CRF(self.num_tags)
        self._cache = None

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

    def call(self, tokens):
        x = self.embed(tokens)
        x = self.bilstm(x)
        decoded, potentials, lens, trans = self.crf(x)
        self._cache = (potentials, lens, trans)
        return decoded

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        pot, lens, trans = self._cache if self._cache is not None else (None, None, None)
        if pot is None:
            _ = self(x)
            pot, lens, trans = self._cache
        return crf_nll_loss(pot, lens, trans, y, sample_weight=sample_weight, reduction="mean")

    def get_build_config(self):
        return {"input_specs": [{"name": "tokens", "dtype": "int32", "shape": (None,)}]}

    def build_from_config(self, cfg):
        tokens = keras.Input(shape=(None,), dtype="int32", name="tokens")
        _ = self(tokens)
        self.built = True


# Step 2: Add length passthrough (TakeFirst) to ensure secondary input participates
@keras.saving.register_keras_serializable(package="Probe", name="Step2_LengthPassthrough")
class TakeFirst(layers.Layer):
    def call(self, xs):
        x, _ = xs
        return x

@keras.saving.register_keras_serializable(package="Probe", name="Step2_BiLSTMCRF_Length")
class Step2_Length(keras.Model):
    def __init__(self, num_tags: int, vocab_size: int, embedding_dim: int = 64, lstm_units: int = 64, **kwargs):
        name = kwargs.pop('name', "step2_length")
        super().__init__(name=name, **kwargs)
        self.num_tags = int(num_tags)
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.lstm_units = int(lstm_units)
        self.embed = layers.Embedding(self.vocab_size + 1, self.embedding_dim, mask_zero=True)
        self.bilstm = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))
        self.length_passthrough = TakeFirst()
        self.crf = CRF(self.num_tags)
        self._cache = None

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

    def call(self, inputs):
        tokens, lengths = inputs
        x = self.embed(tokens)
        x = self.bilstm(x)
        x = self.length_passthrough([x, lengths])
        decoded, potentials, lens, trans = self.crf(x)
        self._cache = (potentials, lens, trans)
        return decoded

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        pot, lens, trans = self._cache if self._cache is not None else (None, None, None)
        if pot is None:
            _ = self(x)
            pot, lens, trans = self._cache
        return crf_nll_loss(pot, lens, trans, y, sample_weight=sample_weight, reduction="mean")

    def get_build_config(self):
        return {"input_specs": [
            {"name": "tokens", "dtype": "int32", "shape": (None,)},
            {"name": "lengths", "dtype": "int32", "shape": (1,)},
        ]}

    def build_from_config(self, cfg):
        tok = keras.Input(shape=(None,), dtype="int32", name="tokens")
        ln = keras.Input(shape=(1,), dtype="int32", name="lengths")
        _ = self([tok, ln])
        self.built = True


# Step 3: Add char channel (TimeDistributed Embedding + BiLSTM) and concat with word embeddings
@keras.saving.register_keras_serializable(package="Probe", name="Step3_BiLSTMCRF_Chars")
class Step3_Chars(keras.Model):
    def __init__(self, num_tags: int, vocab_size: int, char_vocab_size: int, embedding_dim: int = 64, lstm_units: int = 64, char_dim: int = 16, char_lstm: int = 16, **kwargs):
        name = kwargs.pop('name', "step3_chars")
        super().__init__(name=name, **kwargs)
        self.num_tags = int(num_tags)
        self.vocab_size = int(vocab_size)
        self.char_vocab_size = int(char_vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.lstm_units = int(lstm_units)
        self.char_dim = int(char_dim)
        self.char_lstm = int(char_lstm)
        self.word_embed = layers.Embedding(self.vocab_size + 1, self.embedding_dim, mask_zero=True)
        self.char_embed = layers.TimeDistributed(layers.Embedding(self.char_vocab_size + 1, self.char_dim, mask_zero=True), name="char_embeddings")
        self.char_bilstm = layers.TimeDistributed(layers.Bidirectional(layers.LSTM(self.char_lstm, return_sequences=False)), name="chars_rnn")
        self.concat = layers.Concatenate()
        self.bilstm = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))
        self.crf = CRF(self.num_tags)
        self._cache = None

    def get_config(self):
        base = super().get_config()
        base.update({
            "num_tags": self.num_tags,
            "vocab_size": self.vocab_size,
            "char_vocab_size": self.char_vocab_size,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
            "char_dim": self.char_dim,
            "char_lstm": self.char_lstm,
        })
        return base

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

    def call(self, inputs):
        tokens, chars = inputs
        w = self.word_embed(tokens)
        ce = self.char_embed(chars)
        c = self.char_bilstm(ce)
        x = self.concat([w, c])
        x = self.bilstm(x)
        decoded, potentials, lens, trans = self.crf(x)
        self._cache = (potentials, lens, trans)
        return decoded

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        pot, lens, trans = self._cache if self._cache is not None else (None, None, None)
        if pot is None:
            _ = self(x)
            pot, lens, trans = self._cache
        return crf_nll_loss(pot, lens, trans, y, sample_weight=sample_weight, reduction="mean")

    def get_build_config(self):
        return {"input_specs": [
            {"name": "tokens", "dtype": "int32", "shape": (None,)},
            {"name": "chars", "dtype": "int32", "shape": (None, None)},
        ]}

    def build_from_config(self, cfg):
        tok = keras.Input(shape=(None,), dtype="int32", name="tokens")
        ch = keras.Input(shape=(None, None), dtype="int32", name="chars")
        _ = self([tok, ch])
        self.built = True


def run_step1(tmp_root):
    print("=== Step 1: Minimal ===")
    X, Y = make_varlen_dataset(128, 30, 500, 6, seed=1)
    m = Step1_Minimal(num_tags=6, vocab_size=500, embedding_dim=64, lstm_units=64)
    m.compile(optimizer=keras.optimizers.Adam(1e-3))
    m.fit(X, Y, epochs=1, batch_size=32, verbose=0)
    path = save_and_reload(m, X[:8], os.path.join(tmp_root, 'step1'))
    print('OK step1 ->', path)


def run_step2(tmp_root):
    print("=== Step 2: Length passthrough ===")
    X, Y = make_varlen_dataset(128, 30, 500, 6, seed=2)
    lengths = np.sum(X != 0, axis=1).astype(np.int32)[:, None]
    m = Step2_Length(num_tags=6, vocab_size=500, embedding_dim=64, lstm_units=64)
    m.compile(optimizer=keras.optimizers.Adam(1e-3))
    m.fit([X, lengths], Y, epochs=1, batch_size=32, verbose=0)
    path = save_and_reload(m, [X[:8], lengths[:8]], os.path.join(tmp_root, 'step2'))
    print('OK step2 ->', path)


def run_step3(tmp_root):
    print("=== Step 3: Char channel + concat ===")
    X, Y = make_varlen_dataset(128, 30, 500, 6, seed=3)
    # Build per-token char ids (toy): for each token, random length up to 8, pad to 8
    B, T = X.shape
    max_char = 8
    rng = np.random.default_rng(3)
    chars = np.zeros((B, T, max_char), dtype=np.int32)
    for b in range(B):
        for t in range(T):
            if X[b, t] == 0:
                continue
            L = int(rng.integers(1, max_char + 1))
            chars[b, t, :L] = rng.integers(1, 50, size=L)
    m = Step3_Chars(num_tags=6, vocab_size=500, char_vocab_size=50, embedding_dim=64, lstm_units=64, char_dim=16, char_lstm=16)
    m.compile(optimizer=keras.optimizers.Adam(1e-3))
    m.fit([X, chars], Y, epochs=1, batch_size=16, verbose=0)
    path = save_and_reload(m, [X[:8], chars[:8]], os.path.join(tmp_root, 'step3'))
    print('OK step3 ->', path)


def main():
    tmp_root = os.path.join('data/models/sequenceLabelling', 'probe_symbolic_bilstm_crf_subclass')
    os.makedirs(tmp_root, exist_ok=True)
    run_step1(tmp_root)
    run_step2(tmp_root)
    run_step3(tmp_root)
    print('\nAll probe steps OK')


if __name__ == '__main__':
    main()

