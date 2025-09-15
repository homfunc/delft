KerasHub examples

- kerashub_text_classification_demo.py: Small end-to-end demo that uses a KerasHub BERT backbone + tokenizer for text classification.
  - Run:
    - pip install keras-hub huggingface_hub safetensors
    - python examples/kerashub_text_classification_demo.py
- kerashub_sequence_labeling_demo.py: Tiny end-to-end demo using KerasHub BERT + CRF for sequence labeling.
  - Run:
    - pip install keras-hub huggingface_hub safetensors
    - DELFT_USE_KERASHUB=1 python examples/kerashub_sequence_labeling_demo.py
