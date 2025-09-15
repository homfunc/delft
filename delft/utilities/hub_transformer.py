import os
from typing import Any, Dict

import keras

# KerasHub is the Keras 3-native hub interface. We import lazily to make the
# module importable even when KerasHub is not installed, but fail early when used.

def _import_kerashub():
    try:
        # Top-level namespace varies; models and preprocessors are exported from keras_hub.models
        from keras_hub import models as kh_models  # type: ignore
        return kh_models
    except Exception as e:
        raise RuntimeError(
            "KerasHub is required for hub-based transformer models.\n"
            "Please install it: `pip install keras-hub huggingface_hub safetensors`"
        ) from e


class HubTransformer:
    """
    Thin adapter around KerasHub to provide a minimal, DeLFT-compatible interface
    for transformer backbones and their preprocessors.

    Usage:
        hub = HubTransformer(name_or_hf_url)
        backbone = hub.instantiate_backbone()
        preproc = hub.instantiate_preprocessor()
        outputs = backbone(batch_dict)  # {'sequence_output': ..., 'pooled_output': ...} (keys vary)
    """

    def __init__(self, name: str, delft_local_path: str | None = None):
        # name can be an HF id like "hf://google-bert/bert-base-uncased" or a local preset directory
        self.name = name
        self.local_dir_path = delft_local_path
        self._kh = None
        self._backbone = None
        self._preprocessor = None

    def _resolve_preset(self) -> str:
        if self.local_dir_path and os.path.isdir(self.local_dir_path):
            return self.local_dir_path
        # If user passed a bare HF repo id, normalize to hf:// prefix
        if self.name and not self.name.startswith("hf://") and "/" in self.name:
            return f"hf://{self.name}"
        return self.name

    def instantiate_backbone(self):
        if self._backbone is not None:
            return self._backbone
        self._kh = _import_kerashub()
        preset = self._resolve_preset()
        # Heuristic: use the generic Backbone autoclass; KerasHub auto-detects the right family
        Backbone = getattr(self._kh, "Backbone", None)
        if Backbone is None:
            raise RuntimeError("KerasHub does not expose a generic Backbone autoclass.")
        self._backbone = Backbone.from_preset(preset)
        return self._backbone

    def instantiate_preprocessor(self):
        if self._preprocessor is not None:
            return self._preprocessor
        self._kh = self._kh or _import_kerashub()
        preset = self._resolve_preset()
        Preprocessor = getattr(self._kh, "Preprocessor", None)
        if Preprocessor is None:
            # Some model families embed tokenization inside task models; fall back to a tokenizer-aware backbone
            raise RuntimeError("KerasHub Preprocessor autoclass not found. Ensure keras-hub is up to date.")
        self._preprocessor = Preprocessor.from_preset(preset)
        return self._preprocessor

    def get_preprocessor(self):
        return self.instantiate_preprocessor()


class HFCompatBackbone(keras.layers.Layer):
    """
    A small wrapper to present a HF-like call signature to existing DeLFT code.

    Instead of (input_ids, token_type_ids, attention_mask) positional tensors, we accept a dict
    as produced by KerasHub preprocessors, and return a tuple-like or dict-like output with
    at least a sequence representation compatible with token-level tagging.
    """

    def __init__(self, hub: HubTransformer, **kwargs):
        super().__init__(**kwargs)
        self.hub = hub
        self.backbone = None

    def build(self, input_shape):
        if self.backbone is None:
            self.backbone = self.hub.instantiate_backbone()
        super().build(input_shape)

    def call(self, inputs: Dict[str, Any], training=None):
        # KerasHub backbones expect a dict with fields like 'token_ids', 'padding_mask', 'segment_ids'
        outputs = self.backbone(inputs, training=training)
        # Try common keys; standardize to (sequence_output,) tuple for minimal compatibility
        if isinstance(outputs, dict):
            seq = (
                outputs.get("sequence_output")
                or outputs.get("encoder_outputs")
                or outputs.get("hidden_states")
            )
            if seq is not None:
                return (seq,)
        # Fallback: return as-is; upstream code should be adjusted to consume dict
        return outputs
