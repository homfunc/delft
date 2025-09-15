import os, numpy as np, keras
from delft.sequenceLabelling.models import BidLSTM_CRF_SubKerasModel
from delft.sequenceLabelling.config import ModelConfig

# Tiny config
cfg = ModelConfig(
    model_name='tmp_subclass_direct',
    architecture='BidLSTM_CRF_Subclass',
    embeddings_name='dummy-local',
    word_embedding_size=32,
    char_emb_size=16,
    char_lstm_units=8,
    word_lstm_units=16,
    max_char_length=10,
    dropout=0.2,
    recurrent_dropout=0.0,
    batch_size=2,
)

# Data
def make_data():
    X = [["John","lives","in","Paris"], ["Mary","works","at","Acme"], ["I","love","New","York"]]
    Y = [["B-PER","O","O","B-LOC"], ["B-PER","O","O","B-ORG"], ["O","O","B-LOC","I-LOC"]]
    vocab_tag = {t:i for i,t in enumerate(sorted({t for yy in Y for t in yy}))}
    max_len = max(len(x) for x in X)
    # word vectors (random)
    rng = np.random.default_rng(0)
    Xw = np.zeros((len(X), max_len, 32), dtype='float32')
    Xi = np.zeros((len(X), max_len, 10), dtype='int32')
    Li = np.zeros((len(X), 1), dtype='int32')
    Yy = np.zeros((len(X), max_len), dtype='int32')
    for b, (tokens, tags) in enumerate(zip(X,Y)):
        L = len(tokens)
        Li[b,0] = L
        Xw[b,:L,:] = rng.normal(size=(L,32)).astype('float32')
        Xi[b,:L,:] = rng.integers(1, 10, size=(L,10), dtype=np.int32)
        Yy[b,:L] = np.array([vocab_tag[t] for t in tags], dtype=np.int32)
    return (Xw, Xi, Li), Yy, len(vocab_tag)

(Xw, Xi, Li), Yy, ntags = make_data()

m = BidLSTM_CRF_SubKerasModel(num_tags=ntags, config=cfg)
m.compile(optimizer=keras.optimizers.Adam(1e-3))
m.fit([Xw, Xi, Li], Yy, epochs=1, batch_size=2, verbose=0)

out_dir = '/home/m_thing/development/delft/data/models/tmp_direct_subclass'
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, 'model.keras')
m.save(path)
print('saved', path)
loaded = keras.models.load_model(path)
print('loaded ok', type(loaded))
