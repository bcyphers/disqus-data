import numpy as np
import json
from nltk.corpus import stopwords
from sklearn.decomposition import PCA

def load_sim_matrix(forums):
    """
    """
    # load most partisan words
    partisanship = {}
    with open('./data.json') as f:
        data = json.load(f)
        for i in data:
            partisanship[i['tx']] = i['y']

    vocab = np.load('similarity_cache/vocab.npy')

    sorted_words = sorted(vocab, key=partisanship.get, reverse=True)
    sorted_indexes = [vocab.index(w) for w in sorted_words]

    # subsample a million points from each similarity matrix
    sim_mat = np.zeros((len(forums), 1000 * 1000))
    partisan_1k = sorted_indexes[:1000]
    sw = stopwords.words('english')
    common_1k = [i for i in range(len(vocab)) if vocab[i] not in sw][:1000]
    word_pairs = [(vocab[i], vocab[j]) for i in partisan_1k for j in common_1k]

    # load the similarity matrices one at a time and save the subsamples to a
    # local numpy array
    for i, f in enumerate(forums):
        print(('loading similarity matrix for', f))
        fsims = np.load('similarity_cache/%s.npy' % f)
        # take the 1000 most partisan words x the 1000 most common words
        fsims = fsims[np.array(partisan_1k), :]
        fsims = fsims[:, np.array(common_1k)]
        sim_mat[i, :] = fsims.flatten()

    return sim_mat, word_pairs


def load_sim_vector(forums, word, count=5000):
    """
    Load similarity vectors with `word` for each model in `models`
    """
    vocab = np.load('similarity_cache/vocab.npy')
    word_index = list(vocab).index(word)

    sw = stopwords.words('english')
    subsample = np.array([i for i, w in enumerate(vocab)
                          if w not in sw][:count])
    vocab = vocab[subsample]
    sim_mat = np.zeros((len(forums), len(vocab)))

    # load the similarity matrices one at a time and save the vector including
    # `word` for each one
    for i, f in enumerate(forums):
        print(('loading similarity matrix for', f))
        fsims = np.load('similarity_cache/%s.npy' % f)
        arr = fsims[word_index, :]
        sim_mat[i, :] = arr[subsample]

    return sim_mat, vocab


def train_pca(keys, vocab, sim_mat, n_dims=10):
    pca = PCA(n_dims)
    pca_vecs = pca.fit_transform(sim_mat)
    topics = [pca_topic(vocab, pca, i) for i in range(n_dims)]
    for d in range(n_dims):
        var = pca.explained_variance_ratio_[d]
        print(('dimension %d: %.3f of variance' % (d, var)))
        print((topics[d][:20]))
        print((order_by_dim(keys, pca_vecs, d)))
        print()

    return pca, pca_vecs, topics


def order_by_dim(keys, pca_vecs, dim):
    """ Order PCA-transformed samples by a single PCA dimension """
    sort_vecs = sorted(zip(keys, pca_vecs), key=lambda i: i[1][dim])
    return [(k, v[dim]) for k, v in sort_vecs]

def pca_topic(vocab, pca, dim):
    """ Convert a PCA dimension into a (somewhat) understandable format """
    factors = [(val, vocab[ix]) for ix, val in enumerate(pca.components_[dim])]
    return sorted(factors, key=lambda i: abs(i[0]), reverse=True)
