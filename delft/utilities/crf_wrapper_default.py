from __future__ import annotations
import keras
from keras import ops as K
from keras.models import Model
from keras.utils import register_keras_serializable
from keras.saving import serialize_keras_object, deserialize_keras_object

from keras_crf.layers import CRF as KCRF
from keras_crf.losses import (
    CRFNLLHead, CRFDiceHead, CRFJointDiceNLLHead,
    CRFNLLLoss, CRFDiceLoss, CRFJointDiceNLLLoss,
    nll_loss as crf_nll_loss, dice_loss as crf_dice_loss, joint_dice_nll_loss as crf_joint_loss,
)
from keras import ops as _K

@register_keras_serializable(package="DeLFT")
class CRFModelWrapperDefault(Model):
    def __init__(self, base_model: Model | None, num_tags: int | None, name: str = "crf_wrapper",
                 loss_mode: str = "nll", dice_smooth: float = 1.0, joint_nll_weight: float = 0.2,
                 use_kernel: bool = True, use_boundary: bool = False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.base_model = base_model
        self.num_tags = int(num_tags) if num_tags is not None else None
        self.loss_mode = (loss_mode or "nll").lower()
        self.dice_smooth = float(dice_smooth)
        self.joint_nll_weight = float(joint_nll_weight)
        self.use_kernel = bool(use_kernel)
        self.use_boundary = bool(use_boundary)
        # Instantiate the standard CRF layer from keras3-crf with default internal naming
        # Right-padding compatibility: disable boundaries by default (can be overridden with use_boundary=True)
        self.crf = KCRF(self.num_tags if self.num_tags is not None else 1, name="crf", use_kernel=self.use_kernel, use_boundary=self.use_boundary)
        # Cache for CRF internals per forward call to avoid recomputation during compute_loss
        self._crf_cache = None

    def build(self, input_shape):
        if not self.base_model.built:
            self.base_model.build(input_shape)
        crf_input_shape = self.base_model.output_shape
        self.crf.build(crf_input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None, return_crf_internal=False):
        feats = self.base_model(inputs, training=training, mask=mask)
        seq_lengths = None
        if isinstance(inputs, dict) and 'length_input' in inputs:
            seq_lengths = _K.cast(_K.squeeze(inputs['length_input'], axis=-1), 'int32')
        if seq_lengths is not None:
            decoded, potentials, lengths, trans = self.crf(feats, lengths=seq_lengths)
        else:
            decoded, potentials, lengths, trans = self.crf(feats)
        # Cache CRF internals for reuse in compute_loss within the same training step
        try:
            self._crf_cache = (potentials, lengths, trans)
        except Exception:
            self._crf_cache = None
        if return_crf_internal:
            return (potentials, lengths, trans), decoded
        return decoded

    def make_training_model(self, optimizer=None, metrics=None):
        """
        Build a simplified, backend-agnostic CRF training model.
        Inputs: base_model.inputs + tokens + labels
        Outputs:
          - decoded_output: [B,T]
          - crf_loss_value: [B] per-sample loss vector
        """
        base_inputs = list(self.base_model.inputs)
        tokens_in = keras.Input(shape=(None,), dtype="int32", name="tokens")
        labels_in = keras.Input(shape=(None,), dtype="int32", name="labels")

        # Encoder features
        feats = self.base_model(base_inputs)
        if isinstance(feats, (list, tuple)) and len(feats) == 1:
            feats = feats[0]

        # Token mask from tokens (1 for valid, 0 for pad)
        token_mask = keras.layers.Lambda(lambda t: _K.not_equal(t, 0), name="token_mask")(tokens_in)
        # Per-sample valid lengths from mask
        lengths_flat = keras.layers.Lambda(lambda m: _K.sum(_K.cast(m, "int32"), axis=1), name="lengths_flat")(token_mask)

        # CRF forward
        decoded, potentials, lengths_crf, trans = self.crf(feats, mask=token_mask)

        # Loss head selection (per-sample vector)
        loss_mode = (self.loss_mode or 'nll').lower()
        if loss_mode == 'nll':
            loss_vec = CRFNLLHead(name='crf_loss_value')([potentials, labels_in, lengths_flat, trans])
            reducer = CRFNLLLoss()
        elif loss_mode == 'dice':
            loss_vec = CRFDiceHead(smooth=self.dice_smooth, name='crf_loss_value')([potentials, labels_in, lengths_flat, trans])
            reducer = CRFDiceLoss()
        else:
            loss_vec = CRFJointDiceNLLHead(alpha=self.joint_nll_weight, smooth=self.dice_smooth, name='crf_loss_value')([potentials, labels_in, lengths_flat, trans])
            reducer = CRFJointDiceNLLLoss()

        # Decoded output for metrics only (no gradients through Viterbi)
        decoded_named = keras.layers.Lambda(lambda z: _K.stop_gradient(z), name='decoded_output')(decoded)

        def zero_loss(y_true, y_pred):
            return _K.mean(_K.zeros_like(y_pred[..., :1]))

        # Two-head training model
        train_model = keras.Model(inputs=[tokens_in] + base_inputs + [labels_in], outputs=[decoded_named, loss_vec], name='crf_training_model')
        train_model.output_names = ["decoded_output", "crf_loss_value"]
        train_model.compile(
            optimizer=optimizer or keras.optimizers.Adam(1e-3),
            loss={"decoded_output": zero_loss, "crf_loss_value": lambda y_true, y_pred: _K.mean(y_pred)},
            metrics={"decoded_output": metrics or []},
            run_eagerly=False,
        )
        return train_model

    def compute_crf_loss(self, potentials, lengths, trans, y_true, sample_weight=None):
        return crf_nll_loss(potentials, lengths, trans, y_true, sample_weight=sample_weight, reduction="mean")

    def compute_dice_loss(self, potentials, lengths, trans, y_true):
        return crf_dice_loss(potentials, lengths, trans, y_true, smooth=self.dice_smooth, reduction="mean")

    def compute_total_loss(self, potentials, lengths, trans, y_true, sample_weight=None):
        mode = self.loss_mode
        if mode == 'nll':
            return self.compute_crf_loss(potentials, lengths, trans, y_true, sample_weight)
        elif mode == 'dice':
            return self.compute_dice_loss(potentials, lengths, trans, y_true)
        elif mode in ('dice+nll', 'joint'):
            return crf_joint_loss(potentials, lengths, trans, y_true, alpha=self.joint_nll_weight, smooth=self.dice_smooth, reduction="mean", sample_weight=sample_weight)
        else:
            # fallback
            return self.compute_crf_loss(potentials, lengths, trans, y_true, sample_weight)

    # Backend-agnostic training: override compute_loss and let Keras handle gradients per backend
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # Use cached CRF internals if available; otherwise recompute features and CRF
        potentials = lengths = trans = None
        # If call() already computed CRF internals this step, reuse them to avoid recomputation.
        cached = self._crf_cache
        feats = self.base_model(x, training=True)
        # Attempt to pass true lengths during loss computation as well
        seq_lengths = None
        try:
            inps = x
            if isinstance(inps, dict):
                seq_lengths = inps.get('length_input', None)
                if seq_lengths is None:
                    for v in inps.values():
                        shape = getattr(v, 'shape', None)
                        if shape is not None and len(shape) >= 2 and int(shape[-1]) == 1:
                            seq_lengths = v
                            break
            elif isinstance(inps, (list, tuple)) and len(inps) > 0:
                cand = inps[-1]
                shape = getattr(cand, 'shape', None)
                if shape is not None and len(shape) >= 2 and int(shape[-1]) == 1:
                    seq_lengths = cand
                else:
                    for v in inps:
                        shape = getattr(v, 'shape', None)
                        if shape is not None and len(shape) >= 2 and int(shape[-1]) == 1:
                            seq_lengths = v
                            break
            if seq_lengths is not None:
                seq_lengths = _K.cast(_K.squeeze(seq_lengths, axis=-1), 'int32')
        except Exception:
            seq_lengths = None
        potentials = lengths = trans = None
        if cached is not None:
            try:
                potentials, lengths, trans = cached
            except Exception:
                potentials = lengths = trans = None
        if potentials is None or lengths is None or trans is None:
            try:
                if seq_lengths is not None:
                    try:
                        _, potentials, lengths, trans = self.crf(feats, lengths=seq_lengths)
                    except TypeError:
                        _, potentials, lengths, trans = self.crf(feats, sequence_lengths=seq_lengths)
                else:
                    _, potentials, lengths, trans = self.crf(feats)
            except TypeError:
                # Fall back to default behavior without explicit lengths
                _, potentials, lengths, trans = self.crf(feats)
        loss = self.compute_total_loss(potentials, lengths, trans, y, sample_weight)
        if self.losses:
            loss = loss + sum(self.losses)
        # Clear cache after use to avoid stale references across steps
        self._crf_cache = None
        return loss

    # --- Keras serialization ---
    def get_config(self):
        config = super().get_config()
        config.update({
            "base_model": serialize_keras_object(self.base_model),
            "num_tags": self.num_tags,
            "loss_mode": self.loss_mode,
            "dice_smooth": self.dice_smooth,
            "joint_nll_weight": self.joint_nll_weight,
            "use_kernel": self.use_kernel,
            "use_boundary": self.use_boundary,
            "crf_layer": serialize_keras_object(self.crf),
        })
        return config

    @classmethod
    def from_config(cls, config):
        # In Keras 3 SavedModel, nested trackables (like base_model and crf_layer) are revived
        # from the object graph; they may not be present in the user config dict.
        base_model_cfg = config.pop("base_model", None)
        crf_cfg = config.pop("crf_layer", None)
        num_tags = config.pop("num_tags", None)
        loss_mode = config.pop("loss_mode", "nll")
        dice_smooth = config.pop("dice_smooth", 1.0)
        joint_nll_weight = config.pop("joint_nll_weight", 0.2)
        use_kernel = config.pop("use_kernel", True)
        use_boundary = config.pop("use_boundary", False)
        base_model = deserialize_keras_object(base_model_cfg) if base_model_cfg is not None else None
        obj = cls(base_model=base_model, num_tags=num_tags, loss_mode=loss_mode,
                   dice_smooth=dice_smooth, joint_nll_weight=joint_nll_weight,
                   use_kernel=use_kernel, use_boundary=use_boundary, **config)
        return obj

    # Help Keras 3 build variables without running eager data through call()
    def get_build_config(self):
        try:
            inputs = getattr(self.base_model, "inputs", None)
            if inputs:
                input_shape = [tuple(int(d) if d is not None else None for d in inp.shape) for inp in inputs]
                input_dtype = [str(inp.dtype) for inp in inputs]
                return {"input_shape": input_shape, "input_dtype": input_dtype}
        except Exception:
            pass
        return {}

    def build_from_config(self, build_config):
        input_shape = build_config.get("input_shape", None)
        input_dtype = build_config.get("input_dtype", None)
        if not input_shape or self.base_model is None:
            return
        # Create symbolic inputs and run a symbolic forward pass to materialize variables
        sym_inputs = []
        for shape, dtype in zip(input_shape, input_dtype or []):
            # shape is like (None, None, F); Keras Input expects shape without batch dim
            sym_inputs.append(keras.Input(shape=tuple(shape[1:]), dtype=dtype))
        feats = self.base_model(sym_inputs, training=False)
        _ = self.crf(feats)
        # Touch named vars
        _ = (self.crf.trans, getattr(self.crf, 'left_boundary', None), getattr(self.crf, 'right_boundary', None))
        # Nothing to return; variables are created
