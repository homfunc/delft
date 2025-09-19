import keras
from keras import ops as K
from keras.utils import register_keras_serializable


from delft.utilities.crf_wrapper_default import CRFModelWrapperDefault

import numpy as np

'''
Alternative CRF model wrapper for models having a BERT/transformer layer.
Loss is modified to ignore labels corresponding to tokens being special transformer symbols (e.g. SEP,
CL, PAD, ...), but also those introduced sub-tokens.
'''

@register_keras_serializable(package="DeLFT")
class CRFModelWrapperForBERT(CRFModelWrapperDefault):

    def __init__(self, base_model, num_tags: int, name: str = "crf", **kwargs):
        super().__init__(base_model, num_tags, name=name, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None, return_crf_internal=False):
        feats = self.base_model(inputs, training=training, mask=mask)
        # Derive token mask from inputs (attention_mask / padding_mask)
        token_mask = None
        try:
            if isinstance(inputs, dict):
                am = inputs.get('input_attention_mask') or inputs.get('padding_mask') or inputs.get('attention_mask')
                if am is not None:
                    token_mask = K.cast(K.not_equal(am, 0), 'bool')
            else:
                if isinstance(inputs, (list, tuple)) and len(inputs) >= 3:
                    am = inputs[2]
                    token_mask = K.cast(K.not_equal(am, 0), 'bool')
        except Exception:
            token_mask = None
        lengths_local = None
        if token_mask is not None:
            # Per-sample valid sequence lengths
            lengths_local = K.sum(K.cast(token_mask, 'int32'), axis=1)
            decoded, potentials, _, trans = self.crf(feats, mask=token_mask)
            lengths = lengths_local
        else:
            decoded, potentials, lengths, trans = self.crf(feats)
        if return_crf_internal:
            return (potentials, lengths, trans), decoded
        return decoded

    # Backend-agnostic training: mask special tokens inside compute_loss
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # Ensure we have CRF internals
        potentials = lengths = trans = None
        # Try to derive lengths from current inputs regardless of cache state
        token_mask_global = None
        try:
            if isinstance(x, dict):
                am = x.get('input_attention_mask') or x.get('padding_mask') or x.get('attention_mask')
                if am is not None:
                    token_mask_global = K.cast(K.not_equal(am, 0), 'bool')
            else:
                if isinstance(x, (list, tuple)) and len(x) >= 3:
                    am = x[2]
                    token_mask_global = K.cast(K.not_equal(am, 0), 'bool')
        except Exception:
            token_mask_global = None

        # Reuse cached CRF internals if present from call()
        cached = getattr(self, "_crf_cache", None)
        feats = self.base_model(x, training=True)
        # Derive token mask and recompute CRF if needed
        token_mask = token_mask_global
        if cached is not None:
            try:
                potentials, lengths, trans = cached
            except Exception:
                potentials = lengths = trans = None
        if potentials is None or lengths is None or trans is None:
            if token_mask is not None:
                lengths = K.sum(K.cast(token_mask, 'int32'), axis=1)
                _, potentials, _, trans = self.crf(feats, mask=token_mask)
            else:
                _, potentials, lengths, trans = self.crf(feats)
        # Compute loss (CRF handles masking via lengths)
        loss = self.compute_total_loss(potentials, lengths, trans, y, sample_weight)
        if self.losses:
            loss = loss + sum(self.losses)
        # Clear cache after use to avoid stale references across steps
        self._crf_cache = None
        return loss
