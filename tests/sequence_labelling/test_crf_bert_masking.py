import numpy as np
import keras
import pytest

from delft.utilities.crf_wrapper_for_bert import CRFModelWrapperForBERT

@pytest.mark.parametrize("use_mask", [True, False])
def test_crf_bert_compute_loss_handles_lengths(use_mask):
    # Build a tiny base model that returns (B, T, F) features and accepts the usual BERT inputs
    token_ids_in = keras.Input(shape=(None,), dtype="int32", name="input_token")
    token_type_ids_in = keras.Input(shape=(None,), dtype="int32", name="input_token_type")
    attention_mask_in = keras.Input(shape=(None,), dtype="int32", name="input_attention_mask")

    # Features: cast mask to float and project to F dims
    x = keras.layers.Lambda(lambda m: keras.ops.expand_dims(keras.ops.cast(m, "float32"), -1))(attention_mask_in)
    x = keras.layers.Dense(8, activation=None)(x)  # (B,T,8)
    # Tie unused inputs into the graph so Keras considers them connected
    zero_tie = keras.layers.Lambda(lambda a: keras.ops.cast(a, "float32"))
    x = keras.layers.Add()([
        x,
        keras.layers.Lambda(lambda t: 0.0 * keras.ops.expand_dims(keras.ops.cast(t, "float32"), -1))(token_ids_in),
        keras.layers.Lambda(lambda t: 0.0 * keras.ops.expand_dims(keras.ops.cast(t, "float32"), -1))(token_type_ids_in),
    ])
    base = keras.Model(inputs=[token_ids_in, token_type_ids_in, attention_mask_in], outputs=x)

    model = CRFModelWrapperForBERT(base, num_tags=3)
    model.compile(optimizer=keras.optimizers.Adam(1e-2))

    # Fake inputs
    B, T = 2, 5
    token_ids = np.random.randint(10, 100, size=(B, T)).astype("int32")
    token_type_ids = np.zeros_like(token_ids)
    if use_mask:
        attention_mask = np.ones_like(token_ids)
        attention_mask[:, -2:] = 0  # simulate padding
    else:
        attention_mask = np.ones_like(token_ids)

    # Fake labels (sparse int) matching lengths implied by mask
    y = np.random.randint(0, 3, size=(B, T)).astype("int32")

    # Run a training step; compute_loss should not error
    model.train_on_batch([token_ids, token_type_ids, attention_mask], y)
