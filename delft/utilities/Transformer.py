import os
from typing import Any

# Lazy import of transformers to avoid hard dependency at module import time.

def _import_transformers():
    try:
        import transformers  # type: ignore
        return transformers
    except Exception as e:
        raise RuntimeError(
            "Transformers is required for transformer-based models. "
            "Install: `pip install transformers` (and `pip install tf-keras` if using TF)."
        ) from e

TRANSFORMER_CONFIG_FILE_NAME = 'transformer-config.json'
DEFAULT_TRANSFORMER_TOKENIZER_DIR = "transformer-tokenizer"

LOADING_METHOD_LOCAL_MODEL_DIR = "local_model_dir"
LOADING_METHOD_HUGGINGFACE_NAME = "huggingface"
LOADING_METHOD_PLAIN_MODEL = "plain_model"
LOADING_METHOD_DELFT_MODEL = "delft_model"


class Transformer(object):
    """
    Wrapper around a transformer model (pre-trained or fine-tuned).

    Loading priorities:
     1. model saved locally with DeLFT
     2. local directory
     3. plain files (config/weights/vocab)
     4. HF Hub by name
    """

    def __init__(self, name: str, resource_registry: dict = None, delft_local_path: str = None):

        self.bert_preprocessor = None
        self.transformer_config = None
        self.loading_method = None
        self.model = None

        # In case the model is loaded from a local directory
        self.local_dir_path = None

        # In case the weights, config and vocab are specified separately (model vanilla)
        self.local_weights_file = None
        self.local_config_file = None
        self.local_vocab_file = None

        self.name = name

        if delft_local_path:
            self.loading_method = LOADING_METHOD_DELFT_MODEL
            self.local_dir_path = delft_local_path

        self.tokenizer = None

        if resource_registry:
            self.configure_from_registry(resource_registry)

        self.auth_token = None

        # read possible Hugging Face access token to support private models
        # will be None if the key does not exist
        self.auth_token = os.getenv('HF_ACCESS_TOKEN')

    def configure_from_registry(self, resource_registry) -> None:
        """
        Fetch transformer information from the registry and infer the loading method:
            1. if no configuration is provided is using HuggingFace with the provided name
            2. if only the directory is provided it will load the model from that directory
            3. if the weights, config and vocab are provided (as in the vanilla models) then 
               it will load them as BertTokenizer and BertModel
        """

        if self.loading_method == LOADING_METHOD_DELFT_MODEL:
            return

        if 'transformers' in resource_registry:
            filtered_resources = list(
                filter(lambda x: 'name' in x and x['name'] == self.name, resource_registry['transformers']))
            if len(filtered_resources) > 0:
                transformer_configuration = list(filtered_resources)[0]
                if 'model_dir' in transformer_configuration:
                    self.local_dir_path = transformer_configuration['model_dir']
                    self.loading_method = LOADING_METHOD_LOCAL_MODEL_DIR
                else:
                    self.loading_method = LOADING_METHOD_PLAIN_MODEL
                    if "path-config" in transformer_configuration and os.path.isfile(
                            transformer_configuration["path-config"]):
                        self.local_config_file = transformer_configuration["path-config"]
                    else:
                        print("Missing path-config or not a file.")

                    if "path-weights" in transformer_configuration and os.path.isfile(
                            transformer_configuration["path-weights"]) or os.path.isfile(
                        transformer_configuration["path-weights"] + ".data-00000-of-00001"):
                        self.local_weights_file = transformer_configuration["path-weights"]
                    else:
                        print("Missing weights-config or not a file.")

                    if "path-vocab" in transformer_configuration and os.path.isfile(
                            transformer_configuration["path-vocab"]):
                        self.local_vocab_file = transformer_configuration["path-vocab"]
                    else:
                        print("Missing vocab-file or not a file.")
            else:
                self.loading_method = LOADING_METHOD_HUGGINGFACE_NAME
                # print(No configuration for", self.name, "Loading from Hugging face.")
        else:
            self.loading_method = LOADING_METHOD_HUGGINGFACE_NAME
            # print("No configuration for", self.name, "Loading from Hugging face.")

    def init_preprocessor(self, max_sequence_length: int,
                          add_prefix_space: bool = True):
        """
        Load the tokenizer; lazy import transformers.
        """
        transformers = _import_transformers()
        AutoTokenizer = transformers.AutoTokenizer
        AutoConfig = transformers.AutoConfig
        BertTokenizer = getattr(transformers, 'BertTokenizer', None)

        if self.loading_method == LOADING_METHOD_HUGGINGFACE_NAME:
            do_lower_case = None
            if str.lower(self.name).find("uncased") != -1:
                do_lower_case = True
            elif str.lower(self.name).find("cased") != -1:
                do_lower_case = False

            common_kwargs = dict(max_length=max_sequence_length, add_prefix_space=add_prefix_space)
            if do_lower_case is not None:
                common_kwargs["do_lower_case"] = do_lower_case
            if self.auth_token is not None:
                common_kwargs["use_auth_token"] = self.auth_token
            self.tokenizer = AutoTokenizer.from_pretrained(self.name, **common_kwargs)

        elif self.loading_method == LOADING_METHOD_LOCAL_MODEL_DIR:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir_path,
                                                           max_length=max_sequence_length,
                                                           add_prefix_space=add_prefix_space)
        elif self.loading_method == LOADING_METHOD_PLAIN_MODEL and BertTokenizer is not None:
            self.tokenizer = BertTokenizer.from_pretrained(self.local_vocab_file)

        elif self.loading_method == LOADING_METHOD_DELFT_MODEL:
            config_path = os.path.join(".", self.local_dir_path, TRANSFORMER_CONFIG_FILE_NAME)
            self.transformer_config = AutoConfig.from_pretrained(config_path)
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.local_dir_path, DEFAULT_TRANSFORMER_TOKENIZER_DIR), config=self.transformer_config)

    def save_tokenizer(self, output_directory):
        self.tokenizer.save_pretrained(output_directory)

    def instantiate_layer(self, load_pretrained_weights=True) -> Any:
        """
        Instantiate a transformer (TF AutoModel) via transformers; lazy import.
        """
        transformers = _import_transformers()
        TFAutoModel = getattr(transformers, 'TFAutoModel', None)
        TFBertModel = getattr(transformers, 'TFBertModel', None)
        AutoConfig = transformers.AutoConfig
        if TFAutoModel is None:
            raise RuntimeError("TensorFlow-based transformers are not available.")

        if self.loading_method == LOADING_METHOD_HUGGINGFACE_NAME:
            if load_pretrained_weights:
                if self.auth_token is not None:
                    try:
                        transformer_model = TFAutoModel.from_pretrained(self.name, from_pt=True,
                                                                        use_auth_token=self.auth_token)
                    except Exception:
                        transformer_model = TFAutoModel.from_pretrained(self.name, use_auth_token=self.auth_token)
                else:
                    try:
                        transformer_model = TFAutoModel.from_pretrained(self.name, from_pt=True)
                    except Exception:
                        transformer_model = TFAutoModel.from_pretrained(self.name)
                self.transformer_config = transformer_model.config
                return transformer_model
            else:
                config_path = os.path.join(".", self.local_dir_path, TRANSFORMER_CONFIG_FILE_NAME)
                self.transformer_config = AutoConfig.from_pretrained(config_path)
                return TFAutoModel.from_config(self.transformer_config)

        elif self.loading_method == LOADING_METHOD_LOCAL_MODEL_DIR:
            if load_pretrained_weights:
                try:
                    transformer_model = TFAutoModel.from_pretrained(self.local_dir_path, from_pt=True)
                except Exception:
                    transformer_model = TFAutoModel.from_pretrained(self.local_dir_path)
                self.transformer_config = transformer_model.config
                return transformer_model
            else:
                config_path = os.path.join(".", self.local_dir_path, TRANSFORMER_CONFIG_FILE_NAME)
                self.transformer_config = AutoConfig.from_pretrained(config_path)
                return TFAutoModel.from_config(self.transformer_config)

        elif self.loading_method == LOADING_METHOD_PLAIN_MODEL:
            if load_pretrained_weights:
                self.transformer_config = AutoConfig.from_pretrained(self.local_config_file)
                raise NotImplementedError("Plain model loading not implemented; use directory or HF Hub.")
            else:
                config_path = os.path.join(".", self.local_dir_path, TRANSFORMER_CONFIG_FILE_NAME)
                self.transformer_config = AutoConfig.from_pretrained(config_path)
                if TFBertModel is None:
                    raise RuntimeError("TFBertModel not available.")
                return TFBertModel.from_config(self.transformer_config)

        else:
            if load_pretrained_weights:
                transformer_model = TFAutoModel.from_pretrained(self.local_dir_path, from_pt=True)
                self.transformer_config = transformer_model.config
                return transformer_model
            else:
                config_path = os.path.join(".", self.local_dir_path, TRANSFORMER_CONFIG_FILE_NAME)
                self.transformer_config = AutoConfig.from_pretrained(config_path)
                return TFAutoModel.from_config(self.transformer_config)
