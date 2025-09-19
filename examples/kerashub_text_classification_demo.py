#!/usr/bin/env python3
"""
Example: KerasHub BERT text classification (end-to-end)

This example demonstrates how to build a tiny text classification dataset,
load a KerasHub BERT backbone + preprocessor, and train a minimal classifier
with DeLFT's text classification model API.

Requirements:
  pip install keras-hub huggingface_hub safetensors

Run:
  python examples/kerashub_text_classification_demo.py
"""

import os
import numpy as np

from delft.textClassification.config import ModelConfig, TrainingConfig
from delft.textClassification.models import getModel
from delft.textClassification.data_generator import DataGenerator

# Use a well-known BERT preset from HF. You can switch to any compatible model.
TRANSFORMER_NAME = "google-bert/bert-base-uncased"

# Tiny toy dataset (binary classification)
texts = [
    "I absolutely loved this movie!",
    "This is the worst film I have ever seen.",
    "What a fantastic performance by the lead actor.",
    "Terrible plot and bad acting.",
    "A delightful and heartwarming story.",
    "I would not recommend this to anyone.",
]
# Labels: 1 for positive, 0 for negative (as one-hot vector of length 1)
labels = np.array([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]], dtype="float32")

# Split into train/valid
train_texts = texts[:4]
train_labels = labels[:4]
valid_texts = texts[4:]
valid_labels = labels[4:]

# Configure the model
model_config = ModelConfig(
    model_name="demo_kerashub_bert",
    architecture="bert",
    transformer_name=TRANSFORMER_NAME,
    list_classes=["positive"],
    maxlen=64,
    batch_size=2,
)
training_config = TrainingConfig(
    learning_rate=2e-5,
    batch_size=2,
    max_epoch=2,
    patience=1,
    early_stop=False,
)

# Build the model
model = getModel(model_config, training_config)
model.print_summary()

# Generators using KerasHub preprocessor attached to the model
train_gen = DataGenerator(
    train_texts,
    train_labels,
    batch_size=training_config.batch_size,
    maxlen=model_config.maxlen,
    list_classes=model_config.list_classes,
    embeddings=None,
    shuffle=True,
    bert_data=True,
    transformer_tokenizer=model.transformer_tokenizer,
)
valid_gen = DataGenerator(
    valid_texts,
    valid_labels,
    batch_size=training_config.batch_size,
    maxlen=model_config.maxlen,
    list_classes=model_config.list_classes,
    embeddings=None,
    shuffle=False,
    bert_data=True,
    transformer_tokenizer=model.transformer_tokenizer,
)

# Compile and train for a couple epochs
from keras.optimizers import Adam
model.model.compile(optimizer=Adam(learning_rate=training_config.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
model.model.fit(train_gen, epochs=training_config.max_epoch)

# Evaluate
pred = model.model.predict(valid_gen)
print("Predictions on validation:", pred.ravel().tolist())
