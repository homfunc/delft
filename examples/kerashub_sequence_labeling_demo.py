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

# Tiny toy dataset (tokenized sentences + IOB labels)
X = [
    ["John", "lives", "in", "New", "York", "."],
    ["Mary", "works", "at", "Acme", "."],
]
y = [
    ["B-PER", "O", "O", "B-LOC", "I-LOC", "O"],
    ["B-PER", "O", "O", "B-ORG", "O"],
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
    max_epoch=2,
    early_stop=False,
)

# Prepare preprocessor (fits label space)
preprocessor = prepare_preprocessor(X, y, model_config)

# Build model
model = get_model(model_config, preprocessor, ntags=len(preprocessor.vocab_tag))
model.print_summary()

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
)

# Compile and train a couple of epochs
import keras
opt = keras.optimizers.Adam(learning_rate=training_config.learning_rate)
try:
    # If CRF wrapper exposes the training graph, compile is already set; be safe
    model.compile(optimizer=opt)
except Exception:
    pass
model.fit(train_gen, epochs=training_config.max_epoch)

# Predict on the training batch to verify decoding runs
preds = model.predict(train_gen[0][0])
print("Pred sequence lengths:", [len(p) for p in preds])
