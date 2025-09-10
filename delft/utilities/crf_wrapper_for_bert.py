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

    # Backend-agnostic training: mask special tokens inside compute_loss
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # Ensure we have CRF internals
        potentials = lengths = trans = None
        if hasattr(self, "_last_crf_tensors") and isinstance(self._last_crf_tensors, (list, tuple)):
            try:
                potentials, lengths, trans = self._last_crf_tensors
            except Exception:
                potentials = lengths = trans = None
        if potentials is None:
            feats = self.base_model(x, training=True)
            _, potentials, lengths, trans = self.crf(feats)
        # Mask positions where y == 0 (special tokens / padding)
        mask_value = 0
        special_mask = K.not_equal(y, mask_value)
        special_mask = K.cast(special_mask, "float32")
        potentials = potentials * K.expand_dims(special_mask, -1)
        loss = self.compute_total_loss(potentials, lengths, trans, y, sample_weight)
        if self.losses:
            loss = loss + sum(self.losses)
        return loss
