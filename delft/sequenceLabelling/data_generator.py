import numpy as np
import os
from delft.utilities.Utilities import truncate_batch_values, len_until_first_pad
from delft.utilities.numpy import shuffle_triple_with_view

import keras
from delft.sequenceLabelling.preprocess import to_vector_single, to_casing_single, to_vector_simple_with_elmo, \
    Preprocessor, BERTPreprocessor
from delft.utilities.Tokenizer import tokenizeAndFilterSimple


class BaseGenerator(keras.utils.Sequence):
    """
    Abstract class for data generator.

    Generate batch of data to feed sequence labeling model, both for training and prediction.
    
    This generator is for input based on word embeddings. We keep embeddings application outside the 
    model to make it considerably more compact and avoid duplication of embeddings layers.
    """
    def __init__(self, x, y,
                batch_size=24,
                preprocessor: Preprocessor = None,
                bert_preprocessor: BERTPreprocessor = None,
                char_embed_size=25,
                embeddings=None,
                max_sequence_length=None,
                tokenize: bool =False,
                shuffle: bool =True,
                features=None,
                output_input_offsets: bool=False,
                use_chain_crf: bool =False,
                pad_to_max_sequence_length: bool | None = None):
        # self.x and self.y are shuffled view of self.original_x and self.original_y
        self.original_x = self.x = x
        self.original_y = self.y = y
        # features here are optional additional features provided in the case of GROBID input for instance
        self.original_features = self.features = features
        self.preprocessor = preprocessor
        self.bert_preprocessor = bert_preprocessor
        if preprocessor:
            self.labels = preprocessor.vocab_tag
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.char_embed_size = char_embed_size
        self.shuffle = shuffle
        self.tokenize = tokenize
        self.max_sequence_length = max_sequence_length
        # Establish a fixed sequence length for all batches only if explicitly requested
        env_pad = os.environ.get('DELFT_PAD_TO_MAX_SEQ', '0') == '1'
        pad_flag = env_pad if pad_to_max_sequence_length is None else bool(pad_to_max_sequence_length)
        self.fixed_sequence_length = max_sequence_length if pad_flag and max_sequence_length else None
        self.output_input_offsets = output_input_offsets
        self.use_chain_crf = use_chain_crf

    def __len__(self):
        '''
        Give the number of batches per epoch
        '''
        # The number of batches is set so that each training sample is seen at most once per epoch
        if self.original_x is None:
            return 0
        elif (len(self.original_x) % self.batch_size) == 0:
            return int(np.floor(len(self.original_x) / self.batch_size))
        else:
            return int(np.floor(len(self.original_x) / self.batch_size) + 1)

    @property
    def __getitem__(self, index):
        '''
        Generate one batch of data
        '''
        raise NotImplementedError("Subclasses should implement this")

    def on_epoch_end(self):
        '''
        In case we are training, we can shuffle the training data for the next epoch.
        '''
        # If we are predicting, we don't need to shuffle
        if self.original_y is None:
            return

        # shuffle dataset at each epoch
        if self.shuffle:
            self.x, self.y, self.features = shuffle_triple_with_view(self.original_x, self.original_y, self.original_features)

    @property
    def __data_generation(self, index):
        '''
        Generates data containing batch_size samples
        '''
        raise NotImplementedError("Subclasses should implement this")


class DataGenerator(BaseGenerator):
    """    
    This generator is for input based on word embeddings. We keep embeddings application outside the 
    model to make it considerably more compact and avoid duplication of embeddings layers.
    """
    def __init__(self, x, y,
                batch_size=24,
                preprocessor=None,
                bert_preprocessor=None,
                char_embed_size=25,
                embeddings=None,
                max_sequence_length=None,
                tokenize=False,
                shuffle=True,
                features=None,
                output_input_offsets=False,
                use_chain_crf=False,
                pad_to_max_sequence_length: bool | None = None):

        super().__init__(x, y, 
                        batch_size=batch_size, 
                        preprocessor=preprocessor,
                        bert_preprocessor=bert_preprocessor, 
                        char_embed_size=char_embed_size, 
                        embeddings=embeddings, 
                        max_sequence_length=max_sequence_length, 
                        tokenize=tokenize, 
                        shuffle=shuffle, 
                        features=features,
                        output_input_offsets=output_input_offsets,
                        use_chain_crf=use_chain_crf,
                        pad_to_max_sequence_length=pad_to_max_sequence_length)
        self.on_epoch_end()

    def __getitem__(self, index):
        '''
        Generate one batch of data, batch_l always last input, so that it can be used easily by the training scorer
        '''
        batch_x, batch_c, batch_f, batch_a, batch_l, batch_y = self.__data_generation(index)
        if self.preprocessor.return_casing:
            return (batch_x, batch_c, batch_a, batch_l), batch_y
        elif self.preprocessor.return_features:
            return (batch_x, batch_c, batch_f, batch_l), batch_y
        else:
            return (batch_x, batch_c, batch_l), batch_y

    def __data_generation(self, index):
        '''
        Generates data containing batch_size samples
        '''
        max_iter = min(self.batch_size, len(self.original_x)-self.batch_size * index)

        # restrict data to index window
        sub_x = self.x[(index * self.batch_size):(index * self.batch_size) + max_iter]

        # tokenize texts in self.x if not already done
        if self.tokenize:
            x_tokenized = [
                tokenizeAndFilterSimple(text)
                for text in sub_x
            ]
        else:
            x_tokenized = sub_x

        # Use a fixed sequence length for all batches when configured
        if self.fixed_sequence_length is not None:
            max_length_x = int(self.fixed_sequence_length)
            max_length_f = max_length_x
            # Truncate or pad tokens to the fixed length
            x_tokenized = np.asarray(truncate_batch_values(x_tokenized, max_length_x), dtype=object)
        else:
            max_length_f = max_length_x = max((len(tokens) for tokens in x_tokenized))
            if self.max_sequence_length and max_length_x > self.max_sequence_length:
                max_length_x = self.max_sequence_length
                # truncation of sequence at max_sequence_length
                x_tokenized = np.asarray(truncate_batch_values(x_tokenized, self.max_sequence_length), dtype=object)

        # prevent sequence of length 1 alone in a batch (this causes an error in the Chain CRF layer)
        extend = False
        if max_length_x == 1:
            max_length_x += 1
            extend = True

        # generate data
        if self.embeddings and self.embeddings.use_ELMo:
            batch_x = to_vector_simple_with_elmo(x_tokenized, self.embeddings, max_length_x, extend=extend)
        else:
            batch_x = np.zeros((max_iter, max_length_x, self.embeddings.embed_size), dtype='float32')
            for i in range(0, max_iter):
                batch_x[i] = to_vector_single(x_tokenized[i], self.embeddings, max_length_x)

        # store tag embeddings
        batch_y = None
        if self.y is not None:
            # note: tags are always already "tokenized" by input token
            batch_y = self.y[(index*self.batch_size):(index*self.batch_size)+max_iter]
            if self.fixed_sequence_length is not None:
                max_length_y = int(self.fixed_sequence_length)
                batch_y = np.asarray(truncate_batch_values(batch_y, max_length_y), dtype=object)
            else:
                max_length_y = max((len(y_row) for y_row in batch_y))
                if self.max_sequence_length and max_length_y > self.max_sequence_length:
                    # truncation of sequence at max_sequence_length
                    batch_y = np.asarray(truncate_batch_values(batch_y, self.max_sequence_length), dtype=object)

        batch_f = np.zeros((batch_x.shape[0:2]), dtype=np.int32)
        if self.preprocessor.return_features:
            sub_f = self.features[(index * self.batch_size):(index * self.batch_size) + max_iter]
            if self.fixed_sequence_length is not None:
                max_length_f = int(self.fixed_sequence_length)
                sub_f = truncate_batch_values(sub_f, max_length_f)
            else:
                if self.max_sequence_length and max_length_f > self.max_sequence_length:
                    max_length_f = self.max_sequence_length
                    # truncation of sequence at max_sequence_length
                    sub_f = truncate_batch_values(sub_f, self.max_sequence_length)
            batch_f = self.preprocessor.transform_features(sub_f, extend=extend)
        
        batch_a = np.zeros((max_iter, max_length_x), dtype=np.int32)
        if self.preprocessor.return_casing:
            for i in range(0, max_iter):
                batch_a[i] = to_casing_single(x_tokenized[i], max_length_x) 

        if self.y is not None:
            if self.use_chain_crf:
                batches, batch_y = self.preprocessor.transform(x_tokenized, batch_y, extend=extend, label_indices=False)
            else:
                batches, batch_y = self.preprocessor.transform(x_tokenized, batch_y, extend=extend, label_indices=True)
            # Truncate then pad labels to fixed length if configured
            batch_y = np.asarray(truncate_batch_values(batch_y, max_length_x), dtype=np.int32)
            if self.fixed_sequence_length is not None and batch_y.shape[1] != max_length_x:
                By, Ty = batch_y.shape
                pad_y = np.zeros((By, max_length_x), dtype=batch_y.dtype)
                pad_y[:, :Ty] = batch_y
                batch_y = pad_y
        else:
            batches = self.preprocessor.transform(x_tokenized, extend=extend)

        batch_c = np.asarray(batches[0], dtype=np.int32)
        batch_l = batches[1]

        # If using a fixed sequence length, pad char/features to match it
        if self.fixed_sequence_length is not None:
            T_target = int(self.fixed_sequence_length)
            # Pad chars to (B, T_target, max_char)
            if batch_c.shape[1] < T_target:
                Bc, Tc, Cc = batch_c.shape
                pad_c = np.zeros((Bc, T_target, Cc), dtype=batch_c.dtype)
                pad_c[:, :Tc, :] = batch_c
                batch_c = pad_c
            # Pad features to (B, T_target) if present and mismatched
            if isinstance(batch_f, np.ndarray) and batch_f.shape[1] != T_target:
                Bf, Tf = batch_f.shape
                pad_f = np.zeros((Bf, T_target), dtype=batch_f.dtype)
                pad_f[:, :min(Tf, T_target)] = batch_f[:, :min(Tf, T_target)]
                batch_f = pad_f
            # Pad casing to (B, T_target) if present and mismatched
            if self.preprocessor.return_casing and isinstance(batch_a, np.ndarray) and batch_a.shape[1] != T_target:
                Ba, Ta = batch_a.shape
                pad_a = np.zeros((Ba, T_target), dtype=batch_a.dtype)
                pad_a[:, :min(Ta, T_target)] = batch_a[:, :min(Ta, T_target)]
                batch_a = pad_a

        return batch_x, batch_c, batch_f, batch_a, batch_l, batch_y


class DataGeneratorCRFTagger(DataGenerator):
    """
    Generator variant for CRF tagger models that expect inputs {'tokens', 'labels', ...}
    and outputs {'decoded_output', 'crf_loss_value'}.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __getitem__(self, index):
        batch_x, batch_c, batch_f, batch_a, batch_l, batch_y = self._DataGenerator__data_generation(index)
        # Build tokens mask (1 for valid positions, 0 for pad) from lengths
        B, T = batch_x.shape[0], batch_x.shape[1]
        tokens = np.zeros((B, T), dtype=np.int32)
        left_pad = os.environ.get('DELFT_CRF_LEFT_PADDING', '0') == '1'
        for i in range(B):
            L = int(batch_l[i, 0]) if batch_l.ndim == 2 else int(batch_l[i])
            if L > 0:
                if left_pad:
                    start = max(T - L, 0)
                    tokens[i, start:T] = 1
                else:
                    tokens[i, :min(L, T)] = 1
        # Build inputs dict according to the base model inputs used by the architecture
        inputs = {"tokens": tokens, "labels": batch_y}
        if self.preprocessor.return_casing:
            inputs.update({
                'word_input': batch_x,
                'char_input': batch_c,
                'casing_input': batch_a,
                'length_input': batch_l,
            })
        elif self.preprocessor.return_features:
            inputs.update({
                'word_input': batch_x,
                'char_input': batch_c,
                'features_input': batch_f,
                'length_input': batch_l,
            })
        else:
            inputs.update({
                'word_input': batch_x,
                'char_input': batch_c,
                'length_input': batch_l,
            })
        # Targets dict for two-head model
        targets = {
            'decoded_output': batch_y,
            'crf_loss_value': np.zeros((B,), dtype=np.float32)
        }
        return inputs, targets


class DataGeneratorTransformers(BaseGenerator):
    """
    Generate batch of data to feed sequence labeling model, both for training and prediction.
    
    This generator is for input based on transformer embeddings. We keep embeddings application 
    outside the model so that we can serialize the model more easily.  
    """
    def __init__(self, x, y,
                batch_size=24,
                preprocessor=None,
                bert_preprocessor=None,
                char_embed_size=25,
                embeddings=None,
                max_sequence_length=None,
                tokenize=False,
                shuffle=True,
                features=None,
                output_input_offsets=False,
                use_chain_crf=False,
                pad_to_max_sequence_length: bool | None = None):

        super().__init__(x, y, 
                        batch_size=batch_size, 
                        preprocessor=preprocessor, 
                        bert_preprocessor=bert_preprocessor, 
                        char_embed_size=char_embed_size, 
                        embeddings=embeddings, 
                        max_sequence_length=max_sequence_length, 
                        tokenize=tokenize, 
                        shuffle=shuffle, 
                        features=features,
                        output_input_offsets=output_input_offsets,
                        use_chain_crf=use_chain_crf,
                        pad_to_max_sequence_length=pad_to_max_sequence_length)

        if self.bert_preprocessor is not None:
            try:
                if getattr(self.bert_preprocessor, 'empty_features_vector', None) is None:
                    self.bert_preprocessor.empty_features_vector = self.preprocessor.empty_features_vector()
            except Exception:
                pass

        self.on_epoch_end()

    def __getitem__(self, index):
        '''
        Generate one batch of data. Returns a flat dict of tensors to minimize nested structure
        processing in Keras tree utilities (helps Torch Dynamo avoid graph breaks).
        '''
        batch_x, batch_x_types, batch_x_masks, batch_c, batch_f, batch_l, batch_input_offsets, batch_y = self.__data_generation(index)

        inputs = {
            'input_token': batch_x,
            'input_token_type': batch_x_types,
            'input_attention_mask': batch_x_masks,
        }
        # Optional auxiliary inputs (rarely used with transformer-only models)
        if getattr(self.preprocessor, 'return_chars', False):
            inputs['char_input'] = batch_c
        if getattr(self.preprocessor, 'return_features', False):
            inputs['features_input'] = batch_f
        # Lengths are not required when attention_mask is provided, but include if some
        # downstream code expects it
        inputs['length_input'] = batch_l
        # Offsets are only for post-processing; do not include by default in model inputs
        if self.output_input_offsets:
            # Expose offsets under a non-model key for consumers who index the generator directly.
            # Keras will ignore unknown keys when mapping to model inputs.
            inputs['input_offsets'] = batch_input_offsets

        return inputs, batch_y


    def __data_generation(self, index):
        '''
        Generates data containing batch_size samples
        '''
        max_iter = min(self.batch_size, len(self.original_x)-self.batch_size * index)

        # restrict data to index window
        sub_x = self.x[(index * self.batch_size):(index * self.batch_size) + max_iter]

        # tokenize texts in self.x if not already done
        if self.tokenize:
            x_tokenized = [
                tokenizeAndFilterSimple(text)
                for text in sub_x
            ]
        else:
            x_tokenized = sub_x

        # Use a fixed sequence length for all batches when configured
        if self.fixed_sequence_length is not None:
            max_length_x = int(self.fixed_sequence_length)
            max_length_f = max_length_x
            x_tokenized = truncate_batch_values(x_tokenized, max_length_x)
        else:
            max_length_f = max_length_x = max((len(tokens) for tokens in x_tokenized))
            if self.max_sequence_length and max_length_x > self.max_sequence_length:
                max_length_x = self.max_sequence_length
                # truncation of sequence at max_sequence_length
                x_tokenized = truncate_batch_values(x_tokenized, self.max_sequence_length)

        # generate data
        batch_y = None
        
        # tag embeddings
        if self.y is not None:
            # note: tags are always already "tokenized" by input token
            batch_y = self.y[(index*self.batch_size):(index*self.batch_size)+max_iter]
            if self.fixed_sequence_length is not None:
                max_length_y = int(self.fixed_sequence_length)
                batch_y = np.asarray(truncate_batch_values(batch_y, max_length_y), dtype=object)
            else:
                max_length_y = max((len(y_row) for y_row in batch_y))
                if self.max_sequence_length and max_length_y > self.max_sequence_length:
                    # truncation of sequence at max_sequence_length
                     batch_y = np.asarray(truncate_batch_values(batch_y, self.max_sequence_length), dtype=object)

        # features
        if self.preprocessor.return_features:
            sub_f = self.features[(index * self.batch_size):(index * self.batch_size) + max_iter]
            if self.fixed_sequence_length is not None:
                max_length_f = int(self.fixed_sequence_length)
                sub_f = truncate_batch_values(sub_f, max_length_f)
            else:
                if self.max_sequence_length and max_length_f > self.max_sequence_length:
                    max_length_f = self.max_sequence_length
                    # truncation of sequence at max_sequence_length
                    sub_f = truncate_batch_values(sub_f, self.max_sequence_length)
            sub_f = self.preprocessor.transform_features(sub_f)
        else:
            sub_f = None

        # chars and length
        batches = self.preprocessor.transform(x_tokenized)
        batch_c = batches[0]
        batch_l = batches[1]

        # If using a fixed sequence length, pad char/features to match it
        if self.fixed_sequence_length is not None:
            T_target = int(self.fixed_sequence_length)
            # Pad chars to (B, T_target, max_char)
            if self.preprocessor.return_chars and isinstance(batch_c, np.ndarray) and batch_c.shape[1] != T_target:
                Bc, Tc, Cc = batch_c.shape
                pad_c = np.zeros((Bc, T_target, Cc), dtype=batch_c.dtype)
                pad_c[:, :min(Tc, T_target), :] = batch_c[:, :min(Tc, T_target), :]
                batch_c = pad_c
            # Pad features to (B, T_target) if present
            if self.preprocessor.return_features and isinstance(sub_f, np.ndarray) and sub_f.shape[1] != T_target:
                Bf, Tf = sub_f.shape
                pad_f = np.zeros((Bf, T_target), dtype=sub_f.dtype)
                pad_f[:, :min(Tf, T_target)] = sub_f[:, :min(Tf, T_target)]
                sub_f = pad_f

        # to have input as sentence piece token index for transformer layer
        input_ids, token_type_ids, attention_mask, input_chars, input_features, input_labels, input_offsets = self.bert_preprocessor.tokenize_and_align_features_and_labels(
                                                                        x_tokenized, 
                                                                        batch_c,
                                                                        sub_f,
                                                                        batch_y,
                                                                        maxlen=self.max_sequence_length)

        # truncate the batch input vectors for the max length in batch after sub-tokenization
        max_length_x = max((len_until_first_pad(tokens, 0) for tokens in input_ids))

        batch_x = np.asarray(truncate_batch_values(input_ids, max_length_x), dtype=np.int32)
        batch_x_types = np.asarray(truncate_batch_values(token_type_ids, max_length_x), dtype=np.int32)
        batch_x_masks = np.asarray(truncate_batch_values(attention_mask, max_length_x), dtype=np.int32)
        batch_c = np.asarray(truncate_batch_values(input_chars, max_length_x), dtype=np.int32)
        batch_input_offsets = np.asarray(truncate_batch_values(input_offsets, max_length_x), dtype=object)

        if self.y is not None:
            __, batch_y = self.preprocessor.transform(x_tokenized, input_labels, label_indices=True)
            batch_y = np.asarray(truncate_batch_values(batch_y, max_length_x), dtype=np.int32)

        if self.preprocessor.return_features:
            batch_f = np.asarray(truncate_batch_values(input_features, max_length_x), dtype=np.int32)
        else:    
            batch_f = np.zeros((batch_x.shape[0:2]), dtype=np.int32)            

        return batch_x, batch_x_types, batch_x_masks, batch_c, batch_f, batch_l, batch_input_offsets, batch_y

