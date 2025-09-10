#!/usr/bin/env python3
"""
End-to-end smoke test using the Sequence wrapper with BidLSTM_CRF_Subclass.

This script monkey-patches the Embeddings class to provide lightweight random
vectors so no external resources are needed.
"""
import os
import numpy as np

from delft.sequenceLabelling.wrapper import Sequence


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

    # Monkey-patch Embeddings to a lightweight random provider
    import delft.utilities.Embeddings as EmbMod
    class _DummyEmbeddings:
        def __init__(self, name, resource_registry=None, lang='en', extension='vec', use_ELMo=False, use_cache=True, load=True, elmo_model_name=None):
            self.name = name
            self.embed_size = 32
            self.static_embed_size = 32
            self.use_ELMo = False
        def get_word_vector(self, word):
            rng = np.random.default_rng(abs(hash(word)) % (2**32))
            return rng.uniform(-0.5, 0.5, size=(self.embed_size,)).astype('float32')
        def clean_ELMo_cache(self):
            pass
    EmbMod.Embeddings = _DummyEmbeddings

    # Build Sequence wrapper
    seq = Sequence(
        model_name="smoke_sequence_bilstm_crf_subclass",
        architecture="BidLSTM_CRF_Subclass",
        embeddings_name="dummy-local",
        char_emb_size=16,
        max_char_length=10,
        char_lstm_units=8,
        word_lstm_units=16,
        max_sequence_length=16,
        dropout=0.2,
        recurrent_dropout=0.0,
        batch_size=2,
        optimizer='adam',
        learning_rate=1e-3,
        max_epoch=1,
        early_stop=False,
        patience=1,
        fold_number=1,
        multiprocessing=False,
    )

    # Train briefly
    seq.train(X_train, y_train, x_valid=X_valid, y_valid=y_valid)

    # Save model (.keras)
    seq.save(dir_path='data/models/sequenceLabelling/', export_hdf5=False)

    # Validate native Keras load
    import keras
    model_dir = os.path.join('data/models/sequenceLabelling', 'smoke_sequence_bilstm_crf_subclass')
    keras_path = os.path.join(model_dir, 'model.keras')
    loaded = keras.models.load_model(keras_path)
    print('Loaded native model:', type(loaded))
    print('OK')


if __name__ == '__main__':
    main()

