import numpy as np
import keras

from delft.utilities.numpy import shuffle_triple_with_view
from delft.textClassification.preprocess import to_vector_single
from delft.textClassification.preprocess import create_single_input_bert, create_batch_input_bert
from delft.utilities.Tokenizer import tokenizeAndFilterSimple

class DataGenerator(keras.utils.Sequence):
    """
    Generate batch of data to feed text classification model, both for training and prediction.
    For Keras input based on word embeddings, we keep embeddings application outside the model 
    to make it considerably more compact and avoid duplication of embeddings layers.

    When the Keras input will feed a BERT layer, sentence piece tokenization is kept outside 
    the model so that we can serialize the model and have it more compact.  
    """
    def __init__(self, x, y, batch_size=256, maxlen=300, list_classes=[], embeddings=(), shuffle=True, bert_data=False, transformer_tokenizer=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.embeddings = embeddings
        self.list_classes = list_classes
        self.shuffle = shuffle
        self.bert_data = bert_data
        self.transformer_tokenizer = transformer_tokenizer
        self.on_epoch_end()

    def __len__(self):
        """
        Give the number of batches per epoch
        """
        # The number of batches is set so that each training sample is seen at most once per epoch
        if self.x is None:
            return 0
        elif (len(self.x) % self.batch_size) == 0:
            return int(np.floor(len(self.x) / self.batch_size))
        else:
            return int(np.floor(len(self.x) / self.batch_size) + 1)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        batch_x, batch_y = self.__data_generation(index)
        return batch_x, batch_y

    def on_epoch_end(self):
        """
        In case we are training, we can shuffle the training data for the next epoch.
        """
        # If we are predicting, we don't need to shuffle
        if self.y is None:
            return

        # other shuffle dataset for next epoch
        if self.shuffle:
            # Accept Python lists or numpy arrays
            try:
                self.x, self.y, _ = shuffle_triple_with_view(self.x, self.y)
            except Exception:
                import numpy as _np
                x_arr = _np.asarray(self.x, dtype=object)
                y_arr = _np.asarray(self.y, dtype=object)
                x_arr, y_arr, _ = shuffle_triple_with_view(x_arr, y_arr)
                self.x = list(x_arr)
                self.y = list(y_arr)

    def __data_generation(self, index):
        """
        Generates data containing batch_size samples
        """
        max_iter = min(self.batch_size, len(self.x)-self.batch_size*index)

        if not self.bert_data:
            batch_x = np.zeros((max_iter, self.maxlen, self.embeddings.embed_size), dtype='float32')
        else:
            batch_x = None  # Will become a dict of arrays for KerasHub inputs
        batch_y = None
        if self.y is not None:
            batch_y = np.zeros((max_iter, len(self.list_classes)), dtype='float32')

        # Generate data
        if not self.bert_data:
            for i in range(0, max_iter):
                # for input as word embeddings: 
                batch_x[i] = to_vector_single(self.x[(index*self.batch_size)+i], self.embeddings, self.maxlen)
        else:
            # For KerasHub: use its preprocessor to generate token_ids, padding_mask, segment_ids
            # If x contains tokenized lists already, join them back to strings
            normalized = [" ".join(t) if isinstance(t, (list, tuple)) else str(t) for t in self.x[(index*self.batch_size):(index*self.batch_size)+max_iter]]
            if hasattr(self.transformer_tokenizer, '__call__'):
                batch = self.transformer_tokenizer(normalized)
                token_ids = batch.get('token_ids') if 'token_ids' in batch else batch.get('token_ids_0')
                padding_mask = batch.get('padding_mask') if 'padding_mask' in batch else None
                segment_ids = None
                if 'segment_ids' in batch:
                    segment_ids = batch.get('segment_ids')
                elif 'segment_ids_0' in batch:
                    segment_ids = batch.get('segment_ids_0')
                if segment_ids is None and token_ids is not None:
                    segment_ids = [[0]*len(x) for x in token_ids]
                if padding_mask is None and token_ids is not None:
                    padding_mask = [[1]*len(x) for x in token_ids]
            else:
                # Fallback to legacy helpers if a genuine HF tokenizer was passed
                input_ids, input_masks, input_segments = create_batch_input_bert(
                    normalized, maxlen=self.maxlen, transformer_tokenizer=self.transformer_tokenizer)
                token_ids, padding_mask, segment_ids = input_ids, input_masks, input_segments
            # Return dict inputs expected by the KerasHub-based model
            batch_x = {
                'token_ids': np.asarray(token_ids, dtype=np.int32),
                'segment_ids': np.asarray(segment_ids, dtype=np.int32),
                'padding_mask': np.asarray(padding_mask, dtype=np.int32),
            }

        # classes are numerical, so nothing to vectorize for y
        for i in range(0, max_iter):
            if self.y is not None:
                batch_y[i] = self.y[(index*self.batch_size)+i]

        return batch_x, batch_y
