#!/usr/bin/env python3
"""
Example: KerasHub BERT + CRF sequence labeling (end-to-end)

This example shows token-level tagging using a KerasHub BERT backbone and the
DeLFT CRF wrapper. It uses a tiny toy dataset for demonstration.

Run:
  # KerasHub deps
  pip install keras-hub huggingface_hub safetensors
  # Then run
  DELFT_USE_KERASHUB=1 python examples/kerashub_sequence_labeling_demo.py
"""

import os
import numpy as np

from delft.sequenceLabelling.config import ModelConfig, TrainingConfig
from delft.sequenceLabelling.models import get_model
from delft.sequenceLabelling.preprocess import prepare_preprocessor
from delft.sequenceLabelling.data_generator import DataGeneratorTransformers

# Ensure we use the KerasHub path in SL
os.environ.setdefault("DELFT_USE_KERASHUB", "1")

# Choose a BERT preset compatible with KerasHub
TRANSFORMER_NAME = "google-bert/bert-base-uncased"

# A small toy dataset (~20 sentences) with simple PER/LOC/ORG patterns
X = [
    ["John", "lives", "in", "New", "York", "."],
    ["Mary", "works", "at", "Acme", "."],
    ["Alice", "went", "to", "Paris", "."],
    ["Bob", "joined", "Google", "."],
    ["Eve", "moved", "to", "San", "Francisco", "."],
    ["Charlie", "is", "from", "London", "."],
    ["Dave", "visited", "Berlin", "."],
    ["Frank", "studies", "at", "MIT", "."],
    ["Grace", "works", "for", "Amazon", "."],
    ["Heidi", "saw", "New", "Orleans", "."],
    ["Ivan", "met", "Carol", "in", "Rome", "."],
    ["Judy", "joined", "Facebook", "."],
    ["Mallory", "was", "born", "in", "Madrid", "."],
    ["Oscar", "lives", "in", "Boston", "."],
    ["Peggy", "moved", "to", "Tokyo", "."],
    ["Sybil", "studies", "at", "Stanford", "."],
    ["Trent", "visited", "Milan", "."],
    ["Victor", "from", "Dublin", "."],
    ["Walter", "works", "at", "IBM", "."],
    ["Yvonne", "went", "to", "Zurich", "."],
]
y = [
    ["B-PER", "O", "O", "B-LOC", "I-LOC", "O"],
    ["B-PER", "O", "O", "B-ORG", "O"],
    ["B-PER", "O", "O", "B-LOC", "O"],
    ["B-PER", "O", "B-ORG", "O"],
    ["B-PER", "O", "O", "B-LOC", "I-LOC", "O"],
    ["B-PER", "O", "O", "B-LOC", "O"],
    ["B-PER", "O", "B-LOC", "O"],
    ["B-PER", "O", "O", "B-ORG", "O"],
    ["B-PER", "O", "O", "B-ORG", "O"],
    ["B-PER", "O", "B-LOC", "I-LOC", "O"],
    ["B-PER", "O", "B-PER", "O", "B-LOC", "O"],
    ["B-PER", "O", "B-ORG", "O"],
    ["B-PER", "O", "O", "O", "B-LOC", "O"],
    ["B-PER", "O", "O", "B-LOC", "O"],
    ["B-PER", "O", "O", "B-LOC", "O"],
    ["B-PER", "O", "O", "B-ORG", "O"],
    ["B-PER", "O", "B-LOC", "O"],
    ["B-PER", "O", "B-LOC", "O"],
    ["B-PER", "O", "O", "B-ORG", "O"],
    ["B-PER", "O", "O", "B-LOC", "O"],
]

# Build label vocabulary via preprocessor
model_config = ModelConfig(
    model_name="demo_kerashub_bert_crf",
    architecture="BERT_CRF",
    transformer_name=TRANSFORMER_NAME,
    max_sequence_length=64,
    batch_size=2,
    crf_use_boundary=False,
)
training_config = TrainingConfig(
    learning_rate=2e-5,
    batch_size=2,
    max_epoch=20,
    early_stop=False,
)

# Prepare preprocessor (fits label space)
preprocessor = prepare_preprocessor(X, y, model_config)

# Build model
model = get_model(model_config, preprocessor, ntags=len(preprocessor.vocab_tag))
model.print_summary()

# Ensure a KerasHub preprocessor shim is available (in case the model didn't attach one)
if getattr(model, 'transformer_preprocessor', None) is None:
    from delft.utilities.hub_transformer import HubTransformer
    hub = HubTransformer(TRANSFORMER_NAME)
    kh_preproc = hub.get_preprocessor()
    class _DemoShim:
        def __init__(self, kh, p):
            self.kh = kh
            try:
                self.empty_features_vector = p.empty_features_vector()
            except Exception:
                self.empty_features_vector = []
            try:
                self.empty_char_vector = p.empty_char_vector()
            except Exception:
                self.empty_char_vector = []
        def tokenize_and_align_features_and_labels(self, texts, chars, text_features, text_labels, maxlen=512):
            normalized = [" ".join(t) if isinstance(t, (list, tuple)) else str(t) for t in texts]
            batch = self.kh(normalized)
            input_ids = batch['token_ids'] if 'token_ids' in batch else batch.get('token_ids_0')
            if 'segment_ids' in batch:
                token_type_ids = batch['segment_ids']
            elif 'segment_ids_0' in batch:
                token_type_ids = batch['segment_ids_0']
            else:
                token_type_ids = [[0]*len(x) for x in input_ids]
            if 'padding_mask' in batch:
                attention_mask = batch['padding_mask']
                pmask = batch['padding_mask']
            else:
                attention_mask = [[1]*len(x) for x in input_ids]
                pmask = [[1]*len(x) for x in input_ids]
            input_chars = chars
            input_features = text_features
            input_labels = []
            ids_iter = input_ids if input_ids is not None else []
            if text_labels is not None:
                labels_list = text_labels
            else:
                labels_list = [[] for _ in range(len(input_ids) if input_ids is not None else 0)]
            pmask_iter = pmask if pmask is not None else []
            for ids_row, labels_row, mask_row in zip(ids_iter, labels_list, pmask_iter):
                aligned = [0] * len(ids_row)
                try:
                    lr = list(labels_row)
                except Exception:
                    lr = labels_row
                if lr:
                    li = 0
                    for ti in range(len(ids_row)):
                        if bool(mask_row[ti]) and li < len(lr):
                            aligned[ti] = lr[li]
                            li += 1
                input_labels.append(aligned)
            input_offsets = []
            ids_iter2 = input_ids if input_ids is not None else []
            for mask_row, ids_row in zip(pmask, ids_iter2):
                row = []
                for m in mask_row[:len(ids_row)]:
                    row.append((1,1) if m else (0,0))
                if len(row) < len(ids_row):
                    row.extend([(0,0)] * (len(ids_row)-len(row)))
                input_offsets.append(row)
            return input_ids, token_type_ids, attention_mask, input_chars, input_features, input_labels, input_offsets
    model.transformer_preprocessor = _DemoShim(kh_preproc, preprocessor)

# Generator using KerasHub tokenizer attached during init
train_gen = DataGeneratorTransformers(
    X,
    y,
    batch_size=training_config.batch_size,
    preprocessor=preprocessor,
    bert_preprocessor=model.transformer_preprocessor,
    max_sequence_length=model_config.max_sequence_length,
    tokenize=False,
    shuffle=True,
    pad_to_max_sequence_length=True,
)

print(train_gen)
print(train_gen[0])
# print(train_gen[-1])

# Optionally freeze backbone to train CRF faster on tiny data
FREEZE_BACKBONE = True
import keras
# Optional: suppress Torch Dynamo errors to reduce noisy logs when compiling under Torch backend.
try:
    if keras.config.backend() == "torch" and os.environ.get("DELFT_TORCH_SUPPRESS_DYNAMO", "0") == "1":
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
except Exception:
    pass

if FREEZE_BACKBONE:
    base = getattr(model, 'base_model', None)
    if base is not None:
        base.trainable = False
    opt = keras.optimizers.Adam(learning_rate=1e-2)
    try:
        bk = keras.config.backend()
        jit_compile = True if bk == "torch" else "auto"
        # Allow overriding TF eager/graph mode via env (1 to force eager, 0 to allow graph/JIT)
        if bk == "tensorflow":
            run_eagerly = os.environ.get("DELFT_TF_RUN_EAGERLY", "1") == "1"
        else:
            run_eagerly = False
        model.compile(
            optimizer=opt,
            jit_compile=jit_compile,
            run_eagerly=run_eagerly,
        )
    except Exception:
        pass
    model.fit(train_gen, epochs=training_config.max_epoch)
else:
    opt = keras.optimizers.Adam(learning_rate=training_config.learning_rate)
    try:
        bk = keras.config.backend()
        if bk == "tensorflow":
            run_eagerly = os.environ.get("DELFT_TF_RUN_EAGERLY", "1") == "1"
        else:
            run_eagerly = False
        model.compile(optimizer=opt, run_eagerly=run_eagerly)
    except Exception:
        pass
    model.fit(train_gen, epochs=training_config.max_epoch)

# Predict on the training batch to verify decoding runs
batch_inputs, _ = train_gen[0]
preds = model.predict(batch_inputs)

# Pretty-print decoded tags alongside original tokens
# Recompute offsets using the shim to compress subtoken predictions back to word level
dummy_chars = [[model.transformer_preprocessor.empty_char_vector for _ in sent] for sent in X]
_, _, _, _, _, _, input_offsets = model.transformer_preprocessor.tokenize_and_align_features_and_labels(
    X, dummy_chars, None, None, maxlen=model_config.max_sequence_length)

print("\nDecoded tags:\n")
for i, (tokens, y_pred_sub, offsets) in enumerate(zip(X, preds, input_offsets)):
    # Keep only positions marked as valid starts (here offsets marked as (1,1))
    starts = [k for k, off in enumerate(offsets) if tuple(off) == (1, 1)]
    # Compress predictions to first len(tokens) valid starts
    comp = [int(y_pred_sub[k]) for k in starts[:len(tokens)]]
    # Map to tag strings
    tags = preprocessor.inverse_transform(comp)
    print("Sentence", i+1)
    print("Tokens:", tokens)
    print("Tags:  ", tags)
