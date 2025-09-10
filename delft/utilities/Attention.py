from keras import ops as K
from keras import layers, initializers, regularizers, constraints
from keras import backend as Kb


class Attention(layers.Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name=f'{self.name}_W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = int(input_shape[-1])

        if self.bias:
            self.b = self.add_weight(shape=(int(input_shape[1]),),
                                     initializer='zeros',
                                     name=f'{self.name}_b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(
            K.matmul(
                K.reshape(x, (-1, features_dim)),
                K.reshape(self.W, (features_dim, 1))
            ),
            (-1, step_dim)
        )

        if self.bias is not None and self.bias:
            eij = eij + self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a = a * K.cast(mask, a.dtype)

        a = a / (K.sum(a, axis=1, keepdims=True) + K.cast(Kb.epsilon(), a.dtype))

        a = K.expand_dims(a, axis=-1)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.features_dim)


def dot_product(x, kernel):
    """
    Wrapper for dot product operation used in the attention layers
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    return K.squeeze(K.matmul(x, K.expand_dims(kernel, axis=-1)), axis=-1)
