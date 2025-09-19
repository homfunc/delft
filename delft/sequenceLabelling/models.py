import os
import keras

from keras.layers import Dense, LSTM, GRU, Bidirectional, Embedding, Input, Dropout, Reshape
from keras.layers import GlobalMaxPooling1D, TimeDistributed, Conv1D
from keras.layers import Concatenate, Lambda
from keras.models import Model
from keras.models import clone_model
from keras import ops as K
from keras.saving import register_keras_serializable

from delft.sequenceLabelling.config import ModelConfig
from delft.sequenceLabelling.preprocess import Preprocessor, BERTPreprocessor
from delft.utilities.hub_transformer import HubTransformer
from delft.utilities.crf_wrapper_default import CRFModelWrapperDefault
from delft.utilities.crf_wrapper_for_bert import CRFModelWrapperForBERT
from delft.utilities.layers import TakeFirst, ApplyLengthMask, PassMask, ComputeCharLengths, GatherAtIndex

from delft.sequenceLabelling.data_generator import DataGenerator, DataGeneratorTransformers, DataGeneratorCRFTagger
from delft.utilities.Embeddings import load_resource_registry

# CRF layer and losses (Keras 3 backend-agnostic)
from keras_crf.layers import CRF as KCRF
from keras_crf.losses import nll_loss as crf_nll_loss, dice_loss as crf_dice_loss, joint_dice_nll_loss as crf_joint_loss

"""
The sequence labeling models.

Each architecture model is a class implementing a Keras architecture. 
The architecture class can also define the data generator class object to be used, the loss function,
the metrics and the optimizer.
"""


def get_model(config: ModelConfig, preprocessor, ntags=None, load_pretrained_weights=True, local_path=None):
    """
    Return a model instance by its name. This is a facilitator function. 
    """
    print(config.architecture)

    if config.architecture == BidLSTM.name:
        preprocessor.return_word_embeddings = True
        preprocessor.return_chars = True
        preprocessor.return_lengths = True
        return BidLSTM(config, ntags)

    elif config.architecture == BidLSTM_CRF.name:
        preprocessor.return_word_embeddings = True
        preprocessor.return_chars = True
        preprocessor.return_lengths = True
        config.use_crf = True
        return BidLSTM_CRF(config, ntags)


    elif config.architecture == BidLSTM_ChainCRF.name:
        preprocessor.return_word_embeddings = True
        preprocessor.return_chars = True
        preprocessor.return_lengths = True
        config.use_crf = True
        # chain implementation now uses CRF wrapper; do not mark use_chain_crf
        return BidLSTM_ChainCRF(config, ntags)

    elif config.architecture == BidLSTM_CNN.name:
        preprocessor.return_word_embeddings = True
        preprocessor.return_casing = True
        preprocessor.return_chars = True
        preprocessor.return_lengths = True
        return BidLSTM_CNN(config, ntags)

    elif config.architecture == BidLSTM_CNN_CRF.name:
        preprocessor.return_word_embeddings = True
        preprocessor.return_casing = True
        preprocessor.return_chars = True
        preprocessor.return_lengths = True
        config.use_crf = True
        return BidLSTM_CNN_CRF(config, ntags)

    elif config.architecture == BidGRU_CRF.name:
        preprocessor.return_word_embeddings = True
        preprocessor.return_chars = True
        preprocessor.return_lengths = True
        config.use_crf = True
        return BidGRU_CRF(config, ntags)

    elif config.architecture == BidLSTM_CRF_FEATURES.name:
        preprocessor.return_word_embeddings = True
        preprocessor.return_features = True
        preprocessor.return_chars = True
        preprocessor.return_lengths = True
        config.use_crf = True
        return BidLSTM_CRF_FEATURES(config, ntags)

    elif config.architecture == BidLSTM_ChainCRF_FEATURES.name:
        preprocessor.return_word_embeddings = True
        preprocessor.return_features = True
        preprocessor.return_chars = True
        preprocessor.return_lengths = True
        config.use_crf = True
        return BidLSTM_ChainCRF_FEATURES(config, ntags)

    elif config.architecture == BidLSTM_CRF_CASING.name:
        preprocessor.return_word_embeddings = True
        preprocessor.return_casing = True
        preprocessor.return_chars = True
        preprocessor.return_lengths = True
        config.use_crf = True
        return BidLSTM_CRF_CASING(config, ntags)

    elif config.architecture == BERT.name:
        preprocessor.return_bert_embeddings = True
        config.labels = preprocessor.vocab_tag
        return BERT(config, 
                    ntags, 
                    load_pretrained_weights=load_pretrained_weights, 
                    local_path=local_path,
                    preprocessor=preprocessor)

    elif config.architecture == BERT_FEATURES.name:
        preprocessor.return_bert_embeddings = True
        preprocessor.return_features = True
        config.labels = preprocessor.vocab_tag
        return BERT_FEATURES(config, 
                    ntags, 
                    load_pretrained_weights=load_pretrained_weights, 
                    local_path=local_path,
                    preprocessor=preprocessor)

    elif config.architecture == BERT_CRF.name:
        preprocessor.return_bert_embeddings = True
        config.use_crf = True
        config.labels = preprocessor.vocab_tag
        return BERT_CRF(config, 
                        ntags, 
                        load_pretrained_weights=load_pretrained_weights, 
                        local_path=local_path,
                        preprocessor=preprocessor)

    elif config.architecture == BERT_ChainCRF.name:
        preprocessor.return_bert_embeddings = True
        config.use_crf = True
        config.labels = preprocessor.vocab_tag
        return BERT_ChainCRF(config, 
                        ntags, 
                        load_pretrained_weights=load_pretrained_weights, 
                        local_path=local_path,
                        preprocessor=preprocessor)

    elif config.architecture == BERT_CRF_FEATURES.name:
        preprocessor.return_bert_embeddings = True
        preprocessor.return_features = True
        config.use_crf = True
        config.labels = preprocessor.vocab_tag
        return BERT_CRF_FEATURES(config, 
                                ntags, 
                                load_pretrained_weights=load_pretrained_weights, 
                                local_path=local_path,
                                preprocessor=preprocessor)

    elif config.architecture == BERT_ChainCRF_FEATURES.name:
        preprocessor.return_bert_embeddings = True
        preprocessor.return_features = True
        config.use_crf = True
        config.labels = preprocessor.vocab_tag
        return BERT_ChainCRF_FEATURES(config, 
                                ntags, 
                                load_pretrained_weights=load_pretrained_weights, 
                                local_path=local_path,
                                preprocessor=preprocessor)    

    elif config.architecture == BERT_CRF_CHAR.name:
        preprocessor.return_bert_embeddings = True
        preprocessor.return_chars = True
        config.use_crf = True
        config.labels = preprocessor.vocab_tag
        return BERT_CRF_CHAR(config, 
                            ntags,      
                            load_pretrained_weights=load_pretrained_weights, 
                            local_path=local_path,
                            preprocessor=preprocessor)

    elif config.architecture == BERT_CRF_CHAR_FEATURES.name:
        preprocessor.return_bert_embeddings = True
        preprocessor.return_features = True
        preprocessor.return_chars = True
        config.use_crf = True
        config.labels = preprocessor.vocab_tag
        return BERT_CRF_CHAR_FEATURES(config, 
                                    ntags, 
                                    load_pretrained_weights=load_pretrained_weights, 
                                    local_path=local_path,
                                    preprocessor=preprocessor)
    else:
        raise (OSError('Model name does exist: ' + config.architecture))


class BaseModel(object):
    """
    Base class for DeLFT sequence labeling models

    Args:
        config (ModelConfig): DeLFT model configuration object
        ntags (integer): number of different labels of the model
        load_pretrained_weights (boolean): used only when the model contains a transformer layer - indicate whether 
                                           or not we load the pretrained weights of this transformer. For training
                                           a new model set it to True. When getting the full Keras model to load
                                           existing weights, set it False to avoid reloading the pretrained weights. 
        local_path (string): used only when the model contains a transformer layer - the path where to load locally the 
                             pretrained transformer. If None, the transformer model will be fetched from HuggingFace 
                             transformers hub.
    """

    transformer_config = None
    transformer_preprocessor = None

    def __init__(self, config, ntags=None, load_pretrained_weights: bool=True, local_path: str=None, preprocessor=None):
        self.config = config
        self.ntags = ntags
        self.model = None
        self.local_path = local_path
        self.load_pretrained_weights = load_pretrained_weights
        
        self.registry = load_resource_registry("delft/resources-registry.json")

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, batch_size=1)
        return y_pred

    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, batch_size=1)
        return score

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        print('loading model weights', filepath)
        # If a native Keras model file is provided, load it directly
        if filepath.endswith('.keras'):
            from keras.models import load_model
            loaded = load_model(filepath)
            self.model = loaded
            return

        # Try standard loader first
        converted_path = None
        try:
            self.model.load_weights(filepath=filepath)
            converted_path = self._maybe_autoconvert(filepath)
            if converted_path:
                print(f"Converted legacy weights to {converted_path}")
            return
        except Exception as e:
            print('Standard load_weights failed, attempting legacy HDF5 by-name loading:', e)
            try:
                from delft.utilities.weights import load_weights_by_name_from_h5
                load_weights_by_name_from_h5(self.model, filepath)
                converted_path = self._maybe_autoconvert(filepath)
                if converted_path:
                    print(f"Converted legacy weights to {converted_path}")
            except Exception as ee:
                print('Legacy loader failed:', ee)
                raise

    def _maybe_autoconvert(self, legacy_weights_path: str):
        # Save a native Keras model next to the legacy weights for faster future loads
        model_dir = os.path.dirname(legacy_weights_path)
        keras_path = os.path.join(model_dir, 'model.keras')
        if not os.path.exists(keras_path):
            try:
                # Try to symbolically build variables for subclassed models (e.g., CRF wrapper)
                build_cfg_fn = getattr(self.model, 'get_build_config', None)
                build_from_cfg_fn = getattr(self.model, 'build_from_config', None)
                if callable(build_cfg_fn) and callable(build_from_cfg_fn):
                    cfg = build_cfg_fn()
                    build_from_cfg_fn(cfg)
                self.model.save(keras_path)
                return keras_path
            except Exception as e:
                print('Warning: could not auto-convert to .keras:', e)
        return None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def clone_model(self):
        model_copy = clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())
        return model_copy

    def get_generator(self):
        # If model expects 'labels' input (CRF tagger style), use the dedicated generator
        try:
            input_names = [getattr(i, 'name').split(':')[0] for i in self.model.inputs]
            if 'labels' in input_names:
                return DataGeneratorCRFTagger
        except Exception:
            pass
        # default generator
        return DataGenerator

    def print_summary(self):
        base = getattr(self.model, 'base_model', None)
        if base is not None:
            base.summary(expand_nested=True)
        self.model.summary(expand_nested=True)

    def init_transformer(self, config: ModelConfig,
                         load_pretrained_weights: bool,
                         local_path: str,
                         preprocessor: Preprocessor):
        # Always use KerasHub for transformer models
        print(f"Using KerasHub for transformer: {config.transformer_name}")
        hub = HubTransformer(config.transformer_name, delft_local_path=local_path)
        # Preprocessor/tokenizer
        kh_preproc = hub.get_preprocessor()
        # Backbone wrapped to return sequence output compatible with tagging
        from delft.utilities.hub_transformer import HFCompatBackbone
        transformer_model = HFCompatBackbone(hub, name='hub_backbone')

        # Bridge: define a minimal shim that behaves like our BERTPreprocessor
        class _KHPreprocessorShim:
            def __init__(self, kh, preprocessor_obj):
                self.kh = kh
                # supply empty vectors expected by DataGeneratorTransformers
                try:
                    self.empty_features_vector = preprocessor_obj.empty_features_vector()
                except Exception:
                    self.empty_features_vector = []
                try:
                    self.empty_char_vector = preprocessor_obj.empty_char_vector()
                except Exception:
                    self.empty_char_vector = []

            def tokenize_and_align_features_and_labels(self, texts, chars, text_features, text_labels, maxlen=512):
                # KerasHub preprocessors typically accept raw strings; we join tokens when needed
                # If texts are tokenized lists, join them back as space-separated strings
                normalized = [" ".join(t) if isinstance(t, (list, tuple)) else str(t) for t in texts]
                batch = self.kh(normalized)
                # Extract common fields; provide placeholders to match expected tuple structure
                input_ids = batch['token_ids'] if 'token_ids' in batch else batch.get('token_ids_0')
                token_type_ids = None
                if 'segment_ids' in batch:
                    token_type_ids = batch['segment_ids']
                elif 'segment_ids_0' in batch:
                    token_type_ids = batch['segment_ids_0']
                if token_type_ids is None and input_ids is not None:
                    token_type_ids = [[0] * len(x) for x in input_ids]
                attention_mask = batch['padding_mask'] if 'padding_mask' in batch else ([[1] * len(x) for x in input_ids] if input_ids is not None else None)
                # Prepare padding mask array for alignment and offsets
                pmask = batch['padding_mask'] if 'padding_mask' in batch else ([[1] * len(x) for x in input_ids] if input_ids is not None else [])

                # Align labels to sub-tokenization with B/I propagation across sub-tokens.
                input_chars = chars
                input_features = text_features
                input_labels = []
                ids_iter = input_ids if input_ids is not None else []
                labels_list = text_labels if text_labels is not None else ([[]] * (len(input_ids) if input_ids is not None else 0))
                pmask_iter = pmask if pmask is not None else []
                for ids_row, labels_row, mask_row in zip(ids_iter, labels_list, pmask_iter):
                    # valid token positions
                    mask_idx = [k for k, m in enumerate(mask_row[:len(ids_row)]) if bool(m)]
                    # choose first len(labels_row) as word starts
                    try:
                        lr = list(labels_row)
                    except Exception:
                        lr = labels_row
                    starts = mask_idx[:len(lr)] if lr else []
                    # build spans between consecutive starts
                    spans = []
                    for j, s in enumerate(starts):
                        e = mask_idx[mask_idx.index(s) + 1] if (j + 1) < len(starts) else (mask_idx[-1] + 1 if mask_idx else s + 1)
                        spans.append((s, min(e, len(ids_row))))
                    aligned = ["O"] * len(ids_row)
                    for (j, (s, e)) in enumerate(spans):
                        lab = lr[j]
                        if lab == "O":
                            for t in range(s, e):
                                aligned[t] = "O"
                        else:
                            # Normalize to B-/I- form
                            if "-" in lab:
                                pref, typ = lab.split("-", 1)
                            else:
                                pref, typ = "B", lab
                            for t in range(s, e):
                                aligned[t] = ("B-" + typ) if t == s else ("I-" + typ)
                    input_labels.append(aligned)

                # Offsets: mark only word starts (first subtoken) as (1,1), subs/specials as (0,0)
                input_offsets = []
                for labels_row, mask_row, ids_row in zip(labels_list, pmask_iter, ids_iter):
                    row = []
                    mask_slice = mask_row[:len(ids_row)]
                    # When labels are not provided (e.g., during prediction), mark all valid positions as starts
                    if labels_row is None or (hasattr(labels_row, '__len__') and len(labels_row) == 0):
                        for i, m in enumerate(mask_slice):
                            row.append((1,1) if bool(m) else (0,0))
                    else:
                        mask_idx = [k for k, m in enumerate(mask_slice) if bool(m)]
                        try:
                            lab_len = int(len(labels_row))
                        except Exception:
                            lab_len = 0
                        starts = set(mask_idx[:lab_len]) if lab_len > 0 else set()
                        for i, m in enumerate(mask_slice):
                            row.append((1,1) if (bool(m) and i in starts) else (0,0))
                    if len(row) < len(ids_row):
                        row.extend([(0,0)] * (len(ids_row)-len(row)))
                    input_offsets.append(row)
                return input_ids, token_type_ids, attention_mask, input_chars, input_features, input_labels, input_offsets

        # Attach shim to this model instance for downstream generators
        try:
            self.transformer_preprocessor = _KHPreprocessorShim(kh_preproc, preprocessor)
        except Exception:
            self.transformer_preprocessor = None
        self.transformer_config = None
        return transformer_model


class BidLSTM(BaseModel):
    """
    A Keras implementation of simple BidLSTM for sequence labelling with character and word inputs, and softmax final layer.
    """
    name = 'BidLSTM'

    def __init__(self, config, ntags=None):
        super().__init__(config, ntags)

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        # Choose classic or deterministic char path (default: classic)
        import os as _os
        if _os.environ.get('DELFT_DETERMINISTIC_CHAR') == '1':
            # Deterministic, backend-stable path
            char_lengths = ComputeCharLengths(name='char_lengths')(char_input)
            f_seq = TimeDistributed(LSTM(
                config.num_char_lstm_units,
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                unit_forget_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                implementation=1,
                recurrent_dropout=0.0,
            ), name='char_lstm_fwd')(char_embeddings)
            b_seq = TimeDistributed(LSTM(
                config.num_char_lstm_units,
                return_sequences=True,
                go_backwards=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                unit_forget_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                implementation=1,
                recurrent_dropout=0.0,
            ), name='char_lstm_bwd')(char_embeddings)
            f_last = GatherAtIndex(name='char_f_last')([f_seq, K.maximum(char_lengths - 1, 0)])
            b_first = GatherAtIndex(name='char_b_first')([b_seq, K.zeros_like(char_lengths)])
            chars = keras.layers.Concatenate(name='char_repr')([f_last, b_first])
        else:
            # Classic masked last-step via TimeDistributed BiLSTM
            chars = TimeDistributed(
                Bidirectional(LSTM(config.num_char_lstm_units,
                                   return_sequences=False,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',
                                   use_bias=True,
                                   unit_forget_bias=True,
                                   kernel_initializer='glorot_uniform',
                                   recurrent_initializer='orthogonal',
                                   bias_initializer='zeros',
                                   implementation=1,
                                   recurrent_dropout=0.0),
                               name='char_bilstm')
            )(char_embeddings)

        # length of sequence not used by the model, but used by the training scorer
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        x = Concatenate()([word_input, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        # Attach a per-token mask derived from sequence lengths (keeps length_input in graph)
        x = ApplyLengthMask(name="length_mask")([x, length_input])
        pred = Dense(ntags, activation='softmax')(x)

        self.model = Model(inputs=[word_input, char_input, length_input], outputs=[pred])
        #self.model.summary()
        self.config = config


class BidLSTM_CRF(BaseModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """
    name = 'BidLSTM_CRF'

    def __init__(self, config, ntags=None):
        super().__init__(config, ntags)

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        # Classic by default; deterministic path opt-in via env
        import os as _os
        if _os.environ.get('DELFT_DETERMINISTIC_CHAR') == '1':
            # Deterministic, backend-stable path
            char_lengths = ComputeCharLengths(name='char_lengths')(char_input)
            f_seq = TimeDistributed(LSTM(
                config.num_char_lstm_units,
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                unit_forget_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                implementation=1,
                recurrent_dropout=0.0,
            ), name='char_lstm_fwd')(char_embeddings)
            b_seq = TimeDistributed(LSTM(
                config.num_char_lstm_units,
                return_sequences=True,
                go_backwards=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                unit_forget_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                implementation=1,
                recurrent_dropout=0.0,
            ), name='char_lstm_bwd')(char_embeddings)
            f_last = GatherAtIndex(name='char_f_last')([f_seq, K.maximum(char_lengths - 1, 0)])
            b_first = GatherAtIndex(name='char_b_first')([b_seq, K.zeros_like(char_lengths)])
            chars = keras.layers.Concatenate(name='char_repr')([f_last, b_first])
        else:
            # Classic masked last-step via TimeDistributed BiLSTM
            chars = TimeDistributed(
                Bidirectional(LSTM(config.num_char_lstm_units,
                                   return_sequences=False,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',
                                   use_bias=True,
                                   unit_forget_bias=True,
                                   kernel_initializer='glorot_uniform',
                                   recurrent_initializer='orthogonal',
                                   bias_initializer='zeros',
                                   implementation=1,
                                   recurrent_dropout=0.0),
                               name='char_bilstm'),
                name='time_distributed_1')(char_embeddings)

        # length of sequence not used by the model, but used by the training scorer
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        x = Concatenate()([word_input, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        # Attach a per-token mask derived from sequence lengths (keeps length_input in graph)
        x = ApplyLengthMask(name="length_mask")([x, length_input])

        base_model = Model(inputs=[word_input, char_input, length_input], outputs=[x])

        self.model = CRFModelWrapperDefault(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight,
                                            use_kernel=True,
                                            use_boundary=config.crf_use_boundary)
        self.model.build(input_shape=[(None, None, config.word_embedding_size), (None, None, config.max_char_length), (None, None, 1)])
        #self.model.summary()
        self.config = config






class BidLSTM_ChainCRF(BaseModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling with an alternative CRF layer implementation.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """
    name = 'BidLSTM_ChainCRF'

    def __init__(self, config, ntags=None):
        super().__init__(config, ntags)

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=False,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        # Choose classic or deterministic char path (default: classic)
        import os as _os
        if _os.environ.get('DELFT_DETERMINISTIC_CHAR') == '1':
            char_lengths = ComputeCharLengths(name='char_lengths')(char_input)
            f_seq = TimeDistributed(LSTM(
                config.num_char_lstm_units,
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                unit_forget_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                implementation=1,
                recurrent_dropout=0.0,
            ), name='char_lstm_fwd')(char_embeddings)
            b_seq = TimeDistributed(LSTM(
                config.num_char_lstm_units,
                return_sequences=True,
                go_backwards=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                unit_forget_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                implementation=1,
                recurrent_dropout=0.0,
            ), name='char_lstm_bwd')(char_embeddings)
            f_last = GatherAtIndex(name='char_f_last')([f_seq, K.maximum(char_lengths - 1, 0)])
            b_first = GatherAtIndex(name='char_b_first')([b_seq, K.zeros_like(char_lengths)])
            chars = keras.layers.Concatenate(name='char_repr')([f_last, b_first])
        else:
            chars = TimeDistributed(
                Bidirectional(LSTM(config.num_char_lstm_units,
                                   return_sequences=False,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',
                                   use_bias=True,
                                   unit_forget_bias=True,
                                   kernel_initializer='glorot_uniform',
                                   recurrent_initializer='orthogonal',
                                   bias_initializer='zeros',
                                   implementation=1,
                                   recurrent_dropout=0.0)
                ))(char_embeddings)

        # length of sequence not used by the model, but used by the training scorer
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        x = Concatenate()([word_input, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        # Do not pre-project to ntags; the CRF layer will handle projection internally
        # Attach a per-token mask derived from sequence lengths (keeps length_input in graph)
        x = ApplyLengthMask(name="length_mask")([x, length_input])

        base_model = Model(inputs=[word_input, char_input, length_input], outputs=[x])
        self.model = CRFModelWrapperDefault(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight,
                                            use_boundary=config.crf_use_boundary)
        self.model.build(input_shape=[(None, None, config.word_embedding_size), (None, None, config.max_char_length), (None, None, 1)])
        self.config = config


class BidLSTM_CNN(BaseModel):
    """
    A Keras implementation of BidLSTM-CNN for sequence labelling.

    References
    --
    Jason P. C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs". 2016. 
    https://arxiv.org/abs/1511.08308
    """

    name = 'BidLSTM_CNN'

    def __init__(self, config, ntags=None):
        super().__init__(config, ntags)

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding        
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(
                                Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=False,
                                    name='char_embeddings'
                                    ))(char_input)

        dropout = Dropout(config.dropout)(char_embeddings)

        conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
        maxpool_out = TimeDistributed(GlobalMaxPooling1D())(conv1d_out)
        chars = Dropout(config.dropout)(maxpool_out)

        # custom features input and embeddings
        casing_input = Input(batch_shape=(None, None,), dtype='int32', name='casing_input')
        casing_embedding = Embedding(input_dim=config.case_vocab_size,
                           output_dim=config.case_embedding_size,
                           #mask_zero=True,
                           trainable=False,
                           name='casing_embedding')(casing_input)
        casing_embedding = Dropout(config.dropout)(casing_embedding)

        # length of sequence not used by the model, but used by the training scorer
        length_input = Input(batch_shape=(None, 1), dtype='int32')

        # combine words, custom features and characters
        x = Concatenate(axis=-1)([word_input, casing_embedding, chars])
        x = Dropout(config.dropout)(x)
        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        # Ensure length_input participates in the graph while leaving outputs unchanged
        x = TakeFirst(name="length_passthrough")([x, length_input])
        #pred = TimeDistributed(Dense(ntags, activation='softmax'))(x)
        pred = Dense(ntags, activation='softmax')(x)

        self.model = Model(inputs=[word_input, char_input, casing_input, length_input], outputs=[pred])
        self.config = config


class BidLSTM_CNN_CRF(BaseModel):
    """
    A Keras implementation of BidLSTM-CNN-CRF for sequence labelling.

    References
    --
    Xuezhe Ma and Eduard Hovy. "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF". 2016. 
    https://arxiv.org/abs/1603.01354
    """

    name = 'BidLSTM_CNN_CRF'

    def __init__(self, config, ntags=None):
        super().__init__(config, ntags)

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding        
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(
                                Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=False,
                                    name='char_embeddings'
                                    ))(char_input)

        dropout = Dropout(config.dropout)(char_embeddings)

        conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
        maxpool_out = TimeDistributed(GlobalMaxPooling1D())(conv1d_out)
        chars = Dropout(config.dropout)(maxpool_out)

        # custom features input and embeddings
        casing_input = Input(batch_shape=(None, None,), dtype='int32', name='casing_input')

        """
        casing_embedding = Embedding(input_dim=config.case_vocab_size, 
                           output_dim=config.case_embedding_size,
                           mask_zero=True,
                           trainable=False,
                           name='casing_embedding')(casing_input)
        casing_embedding = Dropout(config.dropout)(casing_embedding)
        """

        # length of sequence not used by the model, but used by the training scorer
        length_input = Input(batch_shape=(None, 1), dtype='int32')

        # combine words, custom features and characters
        x = Concatenate(axis=-1)([word_input, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        # Ensure length_input participates in the graph while leaving outputs unchanged
        x = TakeFirst(name="length_passthrough")([x, length_input])
        # Ensure casing_input participates in the graph while leaving outputs unchanged
        x = TakeFirst(name="casing_passthrough")([x, casing_input])

        base_model = Model(inputs=[word_input, char_input, casing_input, length_input], outputs=[x])
        self.model = CRFModelWrapperDefault(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight,
                                            use_boundary=config.crf_use_boundary)
        self.model.build(input_shape=[(None, None, config.word_embedding_size), (None, None, config.max_char_length), (None, None, None), (None, None, 1)])
        self.config = config


class BidGRU_CRF(BaseModel):
    """
    A Keras implementation of BidGRU-CRF for sequence labelling.
    """

    name = 'BidGRU_CRF'

    def __init__(self, config, ntags=None):
        super().__init__(config, ntags)

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        chars = TimeDistributed(Bidirectional(LSTM(config.num_char_lstm_units, return_sequences=False)))(char_embeddings)

        # length of sequence not used by the model, but used by the training scorer
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        x = Concatenate()([word_input, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(GRU(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Bidirectional(GRU(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        # Ensure length_input participates in the graph while leaving outputs unchanged
        x = TakeFirst(name="length_passthrough")([x, length_input])

        base_model = Model(inputs=[word_input, char_input, length_input], outputs=[x])
        self.model = CRFModelWrapperDefault(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight,
                                            use_boundary=config.crf_use_boundary)
        self.model.build(input_shape=[(None, None, config.word_embedding_size), (None, None, config.max_char_length), (None, None, 1)])

        self.config = config


class BidLSTM_CRF_CASING(BaseModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling with additinal features related to casing
    (inferred from word forms).
    """

    name = 'BidLSTM_CRF_CASING'

    def __init__(self, config, ntags=None):
        super().__init__(config, ntags)

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        chars = TimeDistributed(Bidirectional(LSTM(config.num_char_lstm_units, return_sequences=False)))(char_embeddings)

        # custom features input and embeddings
        casing_input = Input(batch_shape=(None, None,), dtype='int32', name='casing_input')

        casing_embedding = Embedding(input_dim=config.case_vocab_size,
                           output_dim=config.case_embedding_size,
                           #mask_zero=True,
                           trainable=False,
                           name='casing_embedding')(casing_input)
        casing_embedding = Dropout(config.dropout)(casing_embedding)

        # length of sequence not used by the model, but used by the training scorer
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        x = Concatenate()([word_input, casing_embedding, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        length_connector = Reshape((-1, 1))(K.cast(length_input, "float32"))
        x = x + 0.0 * length_connector

        base_model = Model(inputs=[word_input, char_input, casing_input, length_input], outputs=[x])
        self.model = CRFModelWrapperDefault(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight,
                                            use_boundary=config.crf_use_boundary)
        self.model.build(input_shape=[(None, None, config.word_embedding_size), (None, None, config.max_char_length), (None, None), (None, None, 1)])
        self.config = config


class BidLSTM_CRF_FEATURES(BaseModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling using tokens combined with 
    additional generic discrete features information.
    """

    name = 'BidLSTM_CRF_FEATURES'

    def __init__(self, config, ntags=None):
        super().__init__(config, ntags)

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        chars = TimeDistributed(Bidirectional(LSTM(config.num_char_lstm_units,
                                                   return_sequences=False)))(char_embeddings)

        # layout features input and embeddings
        features_input = Input(shape=(None, len(config.features_indices)), dtype='float32', name='features_input')

        # The input dimension is calculated by
        # features_vocabulary_size (default 12) * number_of_features + 1 (the zero is reserved for masking / padding)
        features_embedding = TimeDistributed(Embedding(input_dim=config.features_vocabulary_size * len(config.features_indices) + 1,
                                       output_dim=config.features_embedding_size,
                                       # mask_zero=True,
                                       trainable=True,
                                       name='features_embedding'), name="features_embedding_td")(features_input)

        features_embedding_bd = TimeDistributed(Bidirectional(LSTM(config.features_lstm_units, return_sequences=False)),
                                                 name="features_embedding_td_2")(features_embedding)

        features_embedding_out = Dropout(config.dropout)(features_embedding_bd)

        # length of sequence not used by the model, but used by the training scorer
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters, features and word embeddings
        x = Concatenate()([word_input, chars, features_embedding_out])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        x = TakeFirst(name="length_passthrough")([x, length_input])

        base_model = Model(inputs=[word_input, char_input, features_input, length_input], outputs=[x])
        self.model = CRFModelWrapperDefault(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight,
                                            use_boundary=config.crf_use_boundary)
        self.model.build(input_shape=[(None, None, config.word_embedding_size), (None, None, config.max_char_length), (None, None, len(config.features_indices)), (None, None, 1)])
        self.config = config


class BidLSTM_ChainCRF_FEATURES(BaseModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling using tokens combined with 
    additional generic discrete features information and with an alternative CRF layer implementation.
    """

    name = 'BidLSTM_ChainCRF_FEATURES'

    def __init__(self, config, ntags=None):
        super().__init__(config, ntags)

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=False,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        chars = TimeDistributed(Bidirectional(LSTM(config.num_char_lstm_units,
                                                   return_sequences=False)))(char_embeddings)

        # layout features input and embeddings
        features_input = Input(shape=(None, len(config.features_indices)), dtype='float32', name='features_input')

        # The input dimension is calculated by
        # features_vocabulary_size (default 12) * number_of_features + 1 (the zero is reserved for masking / padding)
        features_embedding = TimeDistributed(Embedding(input_dim=config.features_vocabulary_size * len(config.features_indices) + 1,
                                       output_dim=config.features_embedding_size,
                                       # mask_zero=True,
                                       trainable=True,
                                       name='features_embedding'), name="features_embedding_td")(features_input)

        features_embedding_bd = TimeDistributed(Bidirectional(LSTM(config.features_lstm_units, return_sequences=False)),
                                                 name="features_embedding_td_2")(features_embedding)

        features_embedding_out = Dropout(config.dropout)(features_embedding_bd)

        # length of sequence not used by the model, but used by the training scorer
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters, features and word embeddings
        x = Concatenate()([word_input, chars, features_embedding_out])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        # Do not pre-project to ntags; the CRF layer will handle projection internally
        x = TakeFirst(name="length_passthrough")([x, length_input])

        base_model = Model(inputs=[word_input, char_input, features_input, length_input], outputs=[x])
        self.model = CRFModelWrapperDefault(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight,
                                            use_boundary=config.crf_use_boundary)
        self.model.build(input_shape=[(None, None, config.word_embedding_size), (None, None, config.max_char_length), (None, None, len(config.features_indices)), (None, None, 1)])
        self.config = config


class BERT(BaseModel):
    """
    A Keras implementation of BERT for sequence labelling with softmax activation final layer. 

    For training, the BERT layer will be loaded with weights of existing pre-trained BERT model given by the 
    field transformer of the model config (load_pretrained_weights=True).

    For an existing trained model, the BERT layer will be simply initialized (load_pretrained_weights=False),
    without loading pre-trained weights (the weight of the transformer layer will be loaded with the full Keras
    saved model). 

    When initializing the model, we can provide a local_path to load locally the transformer config and (if 
    necessary) the transformer weights. If local_path=None, these files will be fetched from HuggingFace Hub.
    """

    name = 'BERT'

    def __init__(self, config, ntags=None, load_pretrained_weights: bool = True, local_path: str = None, preprocessor=None):
        super().__init__(config, ntags, load_pretrained_weights, local_path)

        transformer_layers = self.init_transformer(config, load_pretrained_weights, local_path, preprocessor)

        input_ids_in = Input(shape=(None,), name='input_token', dtype='int32')
        token_type_ids = Input(shape=(None,), name='input_token_type', dtype='int32')
        attention_mask = Input(shape=(None,), name='input_attention_mask', dtype='int32')

        #embedding_layer = transformer_model(input_ids_in, token_type_ids=token_type_ids)[0]
        embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        embedding_layer = Dropout(0.1)(embedding_layer)
        label_logits = Dense(ntags, activation='softmax')(embedding_layer)

        self.model = Model(inputs=[input_ids_in, token_type_ids, attention_mask], outputs=[label_logits])
        self.config = config

    def get_generator(self):
        return DataGeneratorTransformers

class BERT_FEATURES(BaseModel):
    """
    A Keras implementation of BERT for sequence labelling combined with additional generic discrete features 
    information and with softmax activation final layer. 

    For training, the BERT layer will be loaded with weights of existing pre-trained BERT model given by the 
    field transformer of the model config (load_pretrained_weights=True).

    For an existing trained model, the BERT layer will be simply initialized (load_pretrained_weights=False),
    without loading pre-trained weights (the weight of the transformer layer will be loaded with the full Keras
    saved model). 

    When initializing the model, we can provide a local_path to load locally the transformer config and (if 
    necessary) the transformer weights. If local_path=None, these files will be fetched from HuggingFace Hub.
    """

    name = 'BERT_FEATURES'

    def __init__(self, config, ntags=None, load_pretrained_weights: bool = True, local_path: str = None, preprocessor=None):
        super().__init__(config, ntags, load_pretrained_weights, local_path)

        transformer_layers = self.init_transformer(config, load_pretrained_weights, local_path, preprocessor)

        input_ids_in = Input(shape=(None,), name='input_token', dtype='int32')
        token_type_ids = Input(shape=(None,), name='input_token_type', dtype='int32')
        attention_mask = Input(shape=(None,), name='input_attention_mask', dtype='int32')

        text_embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        text_embedding_layer = Dropout(0.1)(text_embedding_layer)

        # layout features input and embeddings
        features_input = Input(shape=(None, len(config.features_indices)), dtype='float32', name='features_input')

        # The input dimension is calculated by
        # features_vocabulary_size (default 12) * number_of_features + 1 (the zero is reserved for masking / padding)
        features_embedding = TimeDistributed(Embedding(input_dim=config.features_vocabulary_size * len(config.features_indices) + 1,
                                       output_dim=config.features_embedding_size,
                                       # mask_zero=True,
                                       trainable=True,
                                       name='features_embedding'), name="features_embedding_td")(features_input)

        features_embedding_bd = TimeDistributed(Bidirectional(LSTM(config.features_lstm_units, return_sequences=False)),
                                                 name="features_embedding_td_2")(features_embedding)

        features_embedding_out = Dropout(config.dropout)(features_embedding_bd)

        # combine feature and text embeddings
        x = Concatenate()([text_embedding_layer, features_embedding_out])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        label_logits = Dense(ntags, activation='softmax')(x)

        self.model  = Model(inputs=[input_ids_in, features_input, token_type_ids, attention_mask], outputs=[label_logits])
        self.config = config

    def get_generator(self):
        return DataGeneratorTransformers


class BERT_CRF(BaseModel):
    """
    A Keras implementation of BERT-CRF for sequence labelling. The BERT layer will be loaded with weights
    of existing pre-trained BERT model given by the field transformer in the config. 
    """

    name = 'BERT_CRF'

    def __init__(self, config: ModelConfig, ntags=None, load_pretrained_weights:bool =True, local_path: str= None, preprocessor=None):
        super().__init__(config, ntags, load_pretrained_weights, local_path=local_path)

        transformer_layers = self.init_transformer(config, load_pretrained_weights, local_path, preprocessor)

        input_ids_in = Input(shape=(None,), name='input_token', dtype='int32')
        token_type_ids = Input(shape=(None,), name='input_token_type', dtype='int32')
        attention_mask = Input(shape=(None,), name='input_attention_mask', dtype='int32')

        #embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids)[0]
        embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        x = Dropout(0.1)(embedding_layer)

        base_model = Model(inputs=[input_ids_in, token_type_ids, attention_mask], outputs=[x])

        self.model = CRFModelWrapperForBERT(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight)
        self.model.build(input_shape=[(None, None, ), (None, None, ), (None, None, )])
        self.config = config

    def get_generator(self):
        return DataGeneratorTransformers


class BERT_ChainCRF(BaseModel):
    """
    A Keras implementation of BERT-CRF for sequence labelling. The BERT layer will be loaded with weights
    of existing pre-trained BERT model given by the field transformer in the config. 

    This architecture uses an alternative CRF layer implementation.
    """

    name = 'BERT_ChainCRF'

    def __init__(self, config: ModelConfig, ntags=None, load_pretrained_weights:bool=True, local_path: str=None, preprocessor=None):
        super().__init__(config, ntags, load_pretrained_weights, local_path=local_path)

        transformer_layers = self.init_transformer(config, load_pretrained_weights, local_path, preprocessor)

        input_ids_in = Input(shape=(None,), name='input_token', dtype='int32')
        token_type_ids = Input(shape=(None,), name='input_token_type', dtype='int32')
        attention_mask = Input(shape=(None,), name='input_attention_mask', dtype='int32')

        #embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids)[0]
        embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        x = Dropout(0.1)(embedding_layer)

        base_model = Model(inputs=[input_ids_in, token_type_ids, attention_mask], outputs=[x])
        self.model = CRFModelWrapperForBERT(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight)
        self.model.build(input_shape=[(None, None, ), (None, None, ), (None, None, )])
        self.config = config

    def get_generator(self):
        return DataGeneratorTransformers


class BERT_CRF_FEATURES(BaseModel):
    """
    A Keras implementation of BERT-CRF for sequence labelling using tokens combined with 
    additional generic discrete features information. The BERT layer will be loaded with weights
    of existing pre-trained BERT model given by the field transformer in the config. 
    """

    name = 'BERT_CRF_FEATURES'

    def __init__(self, config, ntags=None, load_pretrained_weights=True, local_path:str=None, preprocessor=None):
        super().__init__(config, ntags, load_pretrained_weights, local_path=local_path)

        transformer_layers = self.init_transformer(config, load_pretrained_weights, local_path, preprocessor)

        input_ids_in = Input(shape=(None,), name='input_token', dtype='int32')
        token_type_ids = Input(shape=(None,), name='input_token_type', dtype='int32')
        attention_mask = Input(shape=(None,), name='input_attention_mask', dtype='int32')

        #text_embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids)[0]
        text_embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        text_embedding_layer = Dropout(0.1)(text_embedding_layer)

        # layout features input and embeddings
        features_input = Input(shape=(None, len(config.features_indices)), dtype='float32', name='features_input')

        # The input dimension is calculated by
        # features_vocabulary_size (default 12) * number_of_features + 1 (the zero is reserved for masking / padding)
        features_embedding = TimeDistributed(Embedding(input_dim=config.features_vocabulary_size * len(config.features_indices) + 1,
                                       output_dim=config.features_embedding_size,
                                       # mask_zero=True,
                                       trainable=True,
                                       name='features_embedding'), name="features_embedding_td")(features_input)

        features_embedding_bd = TimeDistributed(Bidirectional(LSTM(config.features_lstm_units, return_sequences=False)),
                                                 name="features_embedding_td_2")(features_embedding)

        features_embedding_out = Dropout(config.dropout)(features_embedding_bd)

        # combine feature and text embeddings
        x = Concatenate()([text_embedding_layer, features_embedding_out])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)

        base_model = Model(inputs=[input_ids_in, features_input, token_type_ids, attention_mask], outputs=[x])

        self.model = CRFModelWrapperForBERT(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight)
        self.model.build(input_shape=[(None, None, ), (None, None, len(config.features_indices)), (None, None, ), (None, None, )])
        self.config = config

    def get_generator(self):
        return DataGeneratorTransformers


class BERT_ChainCRF_FEATURES(BaseModel):
    """
    A Keras implementation of BERT-CRF for sequence labelling using tokens combined with 
    additional generic discrete features information. The BERT layer will be loaded with weights
    of existing pre-trained BERT model given by the field transformer in the config. 

    This architecture uses an alternative CRF layer implementation.
    """

    name = 'BERT_ChainCRF_FEATURES'

    def __init__(self, config, ntags=None, load_pretrained_weights=True, local_path:str=None, preprocessor=None):
        super().__init__(config, ntags, load_pretrained_weights, local_path=local_path)

        transformer_layers = self.init_transformer(config, load_pretrained_weights, local_path, preprocessor)

        input_ids_in = Input(shape=(None,), name='input_token', dtype='int32')
        token_type_ids = Input(shape=(None,), name='input_token_type', dtype='int32')
        attention_mask = Input(shape=(None,), name='input_attention_mask', dtype='int32')

        #text_embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids)[0]
        text_embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        text_embedding_layer = Dropout(0.1)(text_embedding_layer)

        # layout features input and embeddings
        features_input = Input(shape=(None, len(config.features_indices)), dtype='float32', name='features_input')

        # The input dimension is calculated by
        # features_vocabulary_size (default 12) * number_of_features + 1 (the zero is reserved for masking / padding)
        features_embedding = TimeDistributed(Embedding(input_dim=config.features_vocabulary_size * len(config.features_indices) + 1,
                                       output_dim=config.features_embedding_size,
                                       # mask_zero=True,
                                       trainable=True,
                                       name='features_embedding'), name="features_embedding_td")(features_input)

        features_embedding_bd = TimeDistributed(Bidirectional(LSTM(config.features_lstm_units, return_sequences=False)),
                                                 name="features_embedding_td_2")(features_embedding)

        features_embedding_out = Dropout(config.dropout)(features_embedding_bd)

        # combine feature and text embeddings
        x = Concatenate()([text_embedding_layer, features_embedding_out])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)

        base_model = Model(inputs=[input_ids_in, features_input, token_type_ids, attention_mask], outputs=[x])
        self.model = CRFModelWrapperForBERT(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight)
        self.model  = self.model
        self.model.build(input_shape=[(None, None, ), (None, None, len(config.features_indices)), (None, None, ), (None, None, )])
        self.config = config

    def get_generator(self):
        return DataGeneratorTransformers


class BERT_CRF_CHAR(BaseModel):
    """
    A Keras implementation of BERT-CRF for sequence labelling using tokens combined with 
    a character input channel. The BERT layer will be loaded with weights of existing 
    pre-trained BERT model given by the field transformer in the config. 
    """

    name = 'BERT_CRF_CHAR'

    def __init__(self, config, ntags=None, load_pretrained_weights=True, local_path:str=None, preprocessor=None):
        super().__init__(config, ntags, load_pretrained_weights, local_path=local_path)

        transformer_layers = self.init_transformer(config, load_pretrained_weights, local_path, preprocessor)

        input_ids_in = Input(shape=(None,), name='input_token', dtype='int32')
        token_type_ids = Input(shape=(None,), name='input_token_type', dtype='int32')
        attention_mask = Input(shape=(None,), name='input_attention_mask', dtype='int32')

        #text_embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids)[0]
        text_embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        text_embedding_layer = Dropout(0.1)(text_embedding_layer)

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    #mask_zero=True,
                                    trainable=True,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        chars = TimeDistributed(Bidirectional(LSTM(config.num_char_lstm_units,
                                                    return_sequences=False)),
                                                    name="chars_rnn")(char_embeddings)

        # combine characters and word embeddings
        x = Concatenate()([text_embedding_layer, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)

        base_model = Model(inputs=[input_ids_in, char_input, token_type_ids, attention_mask], outputs=[x])
        self.model = CRFModelWrapperForBERT(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight)
        self.model.build(input_shape=[(None, None, ), (None, None, config.max_char_length), (None, None, ), (None, None, )])
        self.config = config

    def get_generator(self):
        return DataGeneratorTransformers


class BERT_CRF_CHAR_FEATURES(BaseModel):
    """
    A Keras implementation of BERT-CRF for sequence labelling using tokens combined with 
    additional generic discrete features information and a character input channel. The 
    BERT layer will be loaded with weights of existing pre-trained BERT model given by 
    the field transformer in the config. 
    """

    name = 'BERT_CRF_CHAR_FEATURES'

    def __init__(self, config, ntags=None, load_pretrained_weights=True, local_path: str= None, preprocessor=None):
        super().__init__(config, ntags, load_pretrained_weights, local_path=local_path)

        transformer_layers = self.init_transformer(config, load_pretrained_weights, local_path, preprocessor)

        input_ids_in = Input(shape=(None,), name='input_token', dtype='int32')
        token_type_ids = Input(shape=(None,), name='input_token_type', dtype='int32')
        attention_mask = Input(shape=(None,), name='input_attention_mask', dtype='int32')

        #text_embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids)[0]
        text_embedding_layer = transformer_layers(input_ids_in, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        text_embedding_layer = Dropout(0.1)(text_embedding_layer)

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    #mask_zero=True,
                                    trainable=True,
                                    #embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
                                    name='char_embeddings'
                                    ))(char_input)

        chars = TimeDistributed(Bidirectional(LSTM(config.num_char_lstm_units,
                                                    return_sequences=False)),
                                                    name="chars_rnn")(char_embeddings)

        # layout features input and embeddings
        features_input = Input(shape=(None, len(config.features_indices)), dtype='float32', name='features_input')

        # The input dimension is calculated by
        # features_vocabulary_size (default 12) * number_of_features + 1 (the zero is reserved for masking / padding)
        features_embedding = TimeDistributed(Embedding(input_dim=config.features_vocabulary_size * len(config.features_indices) + 1,
                                       output_dim=config.features_embedding_size,
                                       # mask_zero=True,
                                       trainable=True,
                                       name='features_embedding'), name="features_embedding_td")(features_input)

        features_embedding_bd = TimeDistributed(Bidirectional(LSTM(config.features_lstm_units, return_sequences=False)),
                                                 name="features_embedding_td_2")(features_embedding)

        features_embedding_out = Dropout(config.dropout)(features_embedding_bd)

        # combine feature, characters and word embeddings
        x = Concatenate()([text_embedding_layer, chars, features_embedding_out])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units,
                               return_sequences=True,
                               recurrent_dropout=config.recurrent_dropout))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)

        base_model = Model(inputs=[input_ids_in, char_input, features_input, token_type_ids, attention_mask], outputs=[x])
        self.model = CRFModelWrapperForBERT(base_model, ntags,
                                            loss_mode=config.crf_loss_type,
                                            dice_smooth=config.crf_dice_smooth,
                                            joint_nll_weight=config.crf_joint_nll_weight)
        self.model.build(input_shape=[(None, None, ), (None, None, config.max_char_length), (None, None, len(config.features_indices)), (None, None, ), (None, None, )])
        self.config = config

    def get_generator(self):
        return DataGeneratorTransformers
