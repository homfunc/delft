# DeLFT CRF training graph migration (Keras 3 + keras-crf)

This release simplifies the CRF training model to closely follow the `keras-crf` example and makes training fully backend-agnostic.

Key changes
- Boundary handling: default `use_boundary=False` for right-padded sequences. For left padding, set `ModelConfig.crf_use_boundary = True`.
- Training graph outputs:
  - `decoded_output`: [B, T]
  - `crf_loss_value`: [B] per-sample CRF loss vector (reduced by mean during training)
- Compile: the decoded head uses a zero loss; the training signal comes from `crf_loss_value`.
- Generator: `DataGeneratorCRFTagger` now supplies a real `tokens` mask (0 pad, 1 valid) and produces targets for `crf_loss_value`.

Compatibility notes
- The former loss head name `crf_log_likelihood_output` is replaced by `crf_loss_value`. The trainer accepts either for backward compatibility, but new models emit `crf_loss_value`.
- Models created with the previous wrapper continue to load; the wrapper serialization now also persists `use_boundary`.

How to switch to left padding (recommended by some models, e.g. Qwen3)
- In your model configuration (ModelConfig), set:
  - `crf_use_boundary = True`
- Ensure your data pipeline produces left-padded sequences (valid tokens right-aligned). When using the CRF wrapper training graph, the `tokens` mask should reflect valid positions accordingly.

Why this change
- Keeps loss computation strictly in-graph using `keras-crf` heads and reducers.
- Improves backend portability (TF / JAX / Torch via Keras 3) and speed (no eager fallback needed).

Action items for downstream projects
- If any training scripts or tests referenced `crf_log_likelihood_output`, update them to `crf_loss_value`.
- Verify that your dataset padding strategy matches your boundary setting:
  - Right padding => `crf_use_boundary=False` (default)
  - Left padding => `crf_use_boundary=True`

CRF loss options (new)
- Sequence labeling CLIs now expose CRF training options:
  - `--crf-loss {nll,dice,dice+nll|joint}`
  - `--crf-dice-smooth FLOAT`
  - `--crf-joint-nll-weight FLOAT`
  - `--crf-use-boundary true|false`
- These flow into ModelConfig and configure the CRF wrapper (keras3-crf), backend-agnostically.

