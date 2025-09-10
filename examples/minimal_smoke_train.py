import numpy as np
import keras
from keras import layers, ops as K
from delft.utilities.crf_wrapper_default import CRFModelWrapperDefault

# Minimal base encoder with safe connectivity to all inputs
D = 50
Cmax = 30
num_tags = 5

word_input = keras.Input(shape=(None, D), name='word_input', dtype='float32')
char_input = keras.Input(shape=(None, Cmax), name='char_input', dtype='int32')
length_input = keras.Input(shape=(1,), name='length_input', dtype='int32')

x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(word_input)
x = layers.Dense(32, activation='tanh')(x)

# Zero-valued connectors to ensure graph connectivity
char_signal = layers.Lambda(lambda c: K.sum(K.cast(c, 'float32'), axis=-1, keepdims=True))(char_input)  # [B,T,1]
zero_char = layers.Lambda(lambda z: z * 0.0)(char_signal)
len_sum = layers.Lambda(lambda l: K.sum(K.cast(l, 'float32')))(length_input)  # scalar
zero_len = layers.Lambda(lambda args: K.zeros_like(args[0]) * args[1])([char_signal, len_sum])  # [B,T,1] * scalar => zeros

x = layers.Add()([x, zero_char, zero_len])
base_model = keras.Model([word_input, char_input, length_input], x)

wrapper = CRFModelWrapperDefault(base_model, num_tags=num_tags, loss_mode='nll', use_boundary=False)
train_model = wrapper.make_training_model()

B, T = 16, 20
rng = np.random.default_rng(7)
lengths = rng.integers(low=5, high=T, size=(B,), dtype=np.int32)

tokens = np.zeros((B, T), dtype=np.int32)
for i, L in enumerate(lengths):
    tokens[i, :int(L)] = 1

word_data = rng.normal(size=(B, T, D)).astype('float32')
char_data = np.zeros((B, T, Cmax), dtype=np.int32)
length_data = lengths.reshape(B, 1)
labels = rng.integers(low=0, high=num_tags, size=(B, T), dtype=np.int32)

x_dict = {
    'tokens': tokens,
    'word_input': word_data,
    'char_input': char_data,
    'length_input': length_data,
    'labels': labels,
}
y_dict = {
    'decoded_output': labels,
    'crf_loss_value': np.zeros((B,), dtype=np.float32),
}

print('Model outputs:', train_model.output_names)
history = train_model.fit(x=x_dict, y=y_dict, epochs=1, batch_size=8, verbose=2)
print('OK: smoke fit completed')
