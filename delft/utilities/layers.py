import keras
from keras.layers import Layer
from keras.utils import register_keras_serializable

@register_keras_serializable(package="DeLFT")
class TakeFirst(Layer):
    """Pass-through layer that returns the first tensor from a list/tuple of inputs.

    This avoids using Lambda with a Python lambda, enabling safe serialization.
    The layer also propagates a mask from one of the inputs to preserve sequence
    lengths for downstream layers (e.g., CRF).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, *args, **kwargs):
        if isinstance(inputs, (list, tuple)):
            return inputs[0]
        return inputs

    def compute_mask(self, inputs, mask=None):
        # Prefer a non-None mask from later inputs (e.g., chars), otherwise fallback to first
        if isinstance(mask, (list, tuple)):
            for i in range(1, len(mask)):
                if mask[i] is not None:
                    return mask[i]
            return mask[0]
        return mask

    def get_config(self):
        return super().get_config()

@register_keras_serializable(package="DeLFT")
class PassMask(Layer):
    """Identity layer that preserves and forwards the input mask.

    Useful to force mask propagation through wrappers like TimeDistributed
    when upstream layers already generated a valid mask (e.g., Embedding(mask_zero=True)).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, *args, **kwargs):
        return inputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

@register_keras_serializable(package="DeLFT")
class ComputeCharLengths(Layer):
    """Compute per-token char lengths from int32 char IDs (0 is pad).
    Input: int32 tensor (B, T, C) -> Output: int32 tensor (B, T)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = False

    def call(self, inputs, *args, **kwargs):
        from keras import ops as K
        x = inputs
        mask = K.cast(K.not_equal(x, 0), 'int32')
        lengths = K.sum(mask, axis=-1)
        return lengths

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

@register_keras_serializable(package="DeLFT")
class GatherAtIndex(Layer):
    """Gather last vector at a given index per token.
    inputs: [seq, index]
      - seq: (B, T, C, H)
      - index: (B, T) int32, with values in [0, C-1]
    output: (B, T, H)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, *args, **kwargs):
        from keras import ops as K
        seq, idx = inputs
        B = K.shape(seq)[0]
        T = K.shape(seq)[1]
        C = K.shape(seq)[2]
        H = K.shape(seq)[3]
        idx = K.maximum(idx, 0)
        idx = K.minimum(idx, C - 1)
        oh = K.one_hot(idx, C)
        oh = K.cast(oh, seq.dtype)
        oh = K.expand_dims(oh, axis=-1)  # (B,T,C,1)
        sel = seq * oh
        out = K.sum(sel, axis=2)
        return out

    def compute_mask(self, inputs, mask=None):
        # Preserve time mask if provided
        if isinstance(mask, (list, tuple)) and len(mask) > 0:
            m = mask[0]
            # reduce char axis if present
            return m[..., 0] if m is not None and len(getattr(m, 'shape', ())) >= 3 else m
        return None

    def compute_output_shape(self, input_shape):
        seq_shape, idx_shape = input_shape
        return (seq_shape[0], seq_shape[1], seq_shape[3])

@register_keras_serializable(package="DeLFT")
class ApplyLengthMask(Layer):
    """Return x unchanged but attach a boolean mask derived from token lengths.

    inputs: [x, lengths] where lengths has shape (batch, 1) or (batch,)
    The mask has shape (batch, timesteps) with True for valid positions.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self._last_mask = None

    def call(self, inputs, *args, **kwargs):
        import keras
        from keras import ops as K
        if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
            return inputs
        x, lengths = inputs[0], inputs[1]
        # shapes: x: (B, T, ...), lengths: (B, 1) or (B,)
        T = K.shape(x)[1]
        rng = K.arange(T)  # (T,)
        lengths = K.cast(K.reshape(lengths, (-1,)), 'int32')  # (B,)
        lengths = K.expand_dims(lengths, axis=-1)  # (B, 1)
        rng = K.expand_dims(rng, axis=0)  # (1, T)
        mask = K.cast(rng < lengths, 'bool')  # (B, T)
        self._last_mask = mask
        return inputs[0]

    def compute_mask(self, inputs, mask=None):
        return self._last_mask

