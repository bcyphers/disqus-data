import csv
import json
import nltk
import os
import pdb
import re
import shutil
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib import colors as mcolors
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from skimage import color
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, \
                                            HashingVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline

from collect_data import *

## Misc ranking functions


def build_link_matrix(data, dedup=True, M=None, N=500, square=False,
                      row_norm=None, col_norm=None, only_en=True, no_channels=True,
                      cutoff=1.5, weight_func='references', src_cats=None,
                      tar_cats=None, sources=None, targets=None):
    """
    Convert the sparse representation that get_edges gives us into a dense
    (directed) transition matrix with normalized rows

    dedup (bool): deduplicate forum names, combine 'duplicates' into one superforum
    M (int): maximum number of source forums to include in the matrix (rows)
        If None, include all forums
    N (int): maximum number of target forums to include in the matrix (columns)
    square (bool): if true, build an MxM square matrix where the column keys are
        the same as the rows (TODO)
    <row, col>_norm ('l1', 'l2', None): how to normalize the rows/columns in
        the matrix. This should be tweaked in the future and is probably
        important to understanding the whole thing
    only_en (bool): if true, only allow english-language forums
    no_channels (bool): if true, exclude channel forums
    cutoff (float): if a forum's outgoing links/person average is below this,
        don't include them
    src_cats (list[str]): if provided, only use forums from these categories as sources
    tar_cats (list[str]): if provided, only use these categories as targets
    src_group (list[str]): if provided, this is the group of sources
    tar_group (list[str]): if provided, this is the group of targets

    returns: MxN matrix of graph edges
    """

    if dedup:
        forum_to_users = data.get_deduped_ftu()
    else:
        forum_to_users = data.forum_to_users

    def forum_out_links(f, cache={}):
        if f not in cache:
            cache[f] = [u for u in forum_to_users[f]
                        if u in data.user_to_forums]
        return cache[f]

    # get the edge graph from the data source. Includes everything we could need
    edges = data.get_forum_edges(dedup)

    # build sources list if necessary
    if sources is None:
        # TODO: is either one of these good enough?
        if weight_func == 'activity':
            weights = data.get_forum_activity(dedup)
        elif weight_func == 'references':
            weights = data.get_weights(dedup)
        else:
            raise ValueError(weight_func)

        # sort all possible sources by our weight metric
        all_sources = sorted(list(edges.keys()), key=lambda f: -weights[f])

        # cull sources that arent the right type
        for f in all_sources[:]:
            if only_en and data.forum_details[f]['language'] != 'en':
                all_sources.remove(f)
            if src_cats and data.forum_details[f]['category'] not in src_cats:
                all_sources.remove(f)
            if no_channels and 'channel-' in f:
                all_sources.remove(f)

        # cull sources that don't meet our cutoff
        sources = []
        for f in all_sources:
            num_users = float(len(forum_out_links(f)))
            connectivity = sum(edges[f].values()) / num_users
            if connectivity < cutoff:
                continue

            # limit to top M forums
            if M is None or len(sources) < M:
                sources.append(f)

    # build targets list if necessary
    if targets is None:
        if square:
            # well that was easy
            targets = sources[:]
        else:
            # generate a mapping of target forums to their occurrences in the graph
            # weighted by number of users for source forum
            weights = defaultdict(float)
            for f, counts in edges.items():
                # only count forums that are in our index
                if f not in sources:
                    continue

                for t, count in counts.items():
                    # do category filtering if necessary
                    category = data.forum_details.get(t, {}).get('category', -1)
                    if tar_cats is None or category in tar_cats:
                        # sort by number of sources that point to each target
                        weights[t] += 1  #float(count) / len(forum_out_links(f))

            # take the top N most-mentioned targets
            targets = sorted(list(weights.keys()), key=lambda t: -weights[t])[:N]

    # M by N matrix
    df = pd.DataFrame(index=sources, columns=targets)

    # iterate over all forums that we have outgoing data for, and finally
    # calculate the edge weight
    for src in sources:
        for tar in targets:
            # number of top users of forum src for whom tar is a top forum
            # "How many top users of src also frequent tar?"
            # think of it like the weight of the edge from src to tar
            users_of_tar = edges[src].get(tar, 0)
            df.ix[src, tar] = float(users_of_tar)

    # normalize the rows
    for f in df.index:
        if row_norm == 'l1':
            # l1 norm produces exact same correlation matrix
            df.ix[f] /= sum(df.ix[f])
        elif row_norm == 'l2':
            df.ix[f] /= np.sqrt(sum(df.ix[f]**2))
        else:
            # default: sort of normalize
            num_users = len(forum_out_links(f))
            df.ix[f] /= num_users

    # normalize columns
    for c in df.columns:
        if col_norm == 'center':
            df[c] -= df[c].mean()
        elif col_norm == 'z':
            df[c] = (df[c] - df[c].mean()) - df[c].std(ddof=0)
        elif col_norm == 'l1':
            df[c] /= sum(df[c])
        elif col_norm == 'l2':
            df[c] /= np.sqrt(sum(df[c]**2))

    print(df.shape)

    return df


def pagerank(df, iters=10):
    for c in df.columns:
        df[c][c] = 0
    A = df.values.T.astype(float)
    n = A.shape[1]
    w, v = linalg.eig(A)
    vec = abs(np.real(v[:n, 0]) / linalg.norm(v[:n, 0], 1))
    ranks = {df.columns[i]: vec[i] for i in range(len(vec))}
    return sorted(list(ranks.items()), key=lambda i: i[1])


def get_correlations(df):
    """
    Generate a correlation matrix for edge graph
    input: DataFrame with columns=variables, index=entities
    output: symmetric correlation mateix with N = len(entities)
    """
    # corrcoeff correlates the *rows* of a dataframe
    return pd.DataFrame(columns=df.index, index=df.index,
                        data=np.corrcoef(df.values.astype(float)))


def top_correlations(cor, forum, n=10):
    cor[forum][forum] = 0
    top = cor[forum].sort_values(ascending=False)[:n]
    for i, f in enumerate(top.index):
        print('%d. %s: %.3f' % (i+1, f, cor[forum][f]))


def print_correlations(df):
    for i, arr in enumerate(np.corrcoef(df.values.astype(float))):
        a = arr[:]
        a[i] = 0
        print(df.columns[i], colored(df.columns[a.argmax()], 'green'), max(a), \
            colored(df.columns[a.argmin()], 'red'), min(a))


def kmeans_cluster(df, n_clusters=10):
    # dataframe: index = entities, columns = variables
    kmeans = KMeans(n_clusters=n_clusters).fit(df.values)
    clusters = []
    for i in range(n_clusters):
        cluster = [df.index[j] for j, label in enumerate(kmeans.labels_)
                   if label == i]
        clusters.append(cluster)

    return clusters


def do_mcl(df, e, r, subset=None):
    # perform the Markov Cluster Algorithm (MCL) with expansion power parameter
    # e and inflation parameter r
    # higher r -> more granular clusters
    # based on
    # https://www.cs.ucsb.edu/~xyan/classes/CS595D-2009winter/MCL_Presentation2.pdf
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].map(lambda v: v**2 if v > 0 else 0)
        if not sum(df[c]):
            print(c, 'has no connections!')
            continue
        df[c] /= sum(df[c])     # column normalize

    if subset:
        df = df[subset].ix[subset]
        for c in df.columns:
            df[c] /= sum(df[c])

    mat = df.values.astype(float)

    converged = False
    while not converged:
        # expand
        last_mat = mat.copy()
        mat = linalg.matrix_power(mat, e)

        # inflate
        for i in range(mat.shape[0]):
            mat[i] **= r
            mat[i] /= sum(mat[i])

            for j in range(mat.shape[1]):
                if mat[i, j] < 1.0e-100:
                    mat[i, j] = 0

        # converge?
        if np.all(last_mat == mat):
            converged = True

    clusters = {}
    scores = {}
    for i in range(mat.shape[1]):
        if sum(mat[:,i]) > 0:
            # find the forums that cluster around this one, sort alphabetically
            cluster = [j for j in range(mat.shape[0]) if mat[j,i] > 0]
            cluster = sorted([df.columns[k] for k in cluster])
            clusters[df.columns[i]] = cluster

    return clusters


def plot_forums(matrix, groups, method='pca', n_groups=None, do_legend=False):
    """
    Perform dimensionality reduction on an N by N matrix and plot it by group
    Dimensionality reduction method can be 'pca' or 'tsne'
    """
    if method == 'pca':
        pca = PCA(n_components=2).fit_transform(matrix)
        df = pd.DataFrame(index=matrix.index, data=pca)
    elif method == 'tsne':
        pca = PCA(n_components=25).fit_transform(matrix)
        tsne = TSNE(perplexity=25, random_state=0).fit_transform(pca)
        df = pd.DataFrame(index=matrix.index, data=tsne)

    legend = []
    n_groups = n_groups or len(groups)
    sort_groups = sorted(list(groups.items()), key=lambda i: -len(i[1]))[:n_groups]

    min_x, max_x = min(df[0]), max(df[0])
    min_y, max_y = min(df[1]), max(df[1])
    center_x = min_x + (max_x - min_x) / 2
    center_y = min_y + (max_y - min_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    # loop over groups of forums
    for i, (f, group) in enumerate(sort_groups):

        # get coordinates of the forum everything's clustered around (the sink)
        sink = (df[0][f], df[1][f])

        # get the geometric center of the group to generate a color
        group_x = (center_x - sum(df[0][g] for g in group) / len(group)) / width
        group_y = (center_y - sum(df[1][g] for g in group) / len(group)) / height
        clr = color.lab2rgb([[[60, 128 * group_x, 128 * group_y]]])[0][0]

        for g in group:
            # get the coordinates of the current forum
            pt = (df[0][g], df[1][g])

            # plot it
            c = plt.scatter(pt[0], pt[1], color=clr)

            # draw a line from the sink to the current point
            if f != g:
                l = plt.plot((sink[0], pt[0]), (sink[1], pt[1]), color=clr)

        legend.append((c, f))

    if do_legend:
        plt.legend(*list(zip(*legend)))
    plt.show()

