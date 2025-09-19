import os
import numpy as np
import keras
from keras import ops as K
from keras_crf import CRF

print('keras_version', keras.__version__)
print('backend', keras.backend.backend())

# Build model with use_kernel=False, feature dim equals units
inp = keras.Input(shape=(None, 4), name='feat')
crf = CRF(units=4, use_kernel=False, use_boundary=True)
decoded, potentials, lens, trans = crf(inp)
model = keras.Model(inp, decoded)

# Eager pass
x = np.random.randn(3, 5, 4).astype(np.float32)
y_before = model(x)

# Save
save_dir = '/home/m_thing/development/delft/data/models/tmp_crf_symbolic_nokernel'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'model.keras')
model.save(model_path)
print('saved_to', model_path)

# Load and compare
loaded = keras.saving.load_model(model_path)
y_after = loaded(x)
np.testing.assert_allclose(K.convert_to_numpy(y_before), K.convert_to_numpy(y_after), rtol=1e-6, atol=1e-6)
print('load_roundtrip_inference_match', True)
