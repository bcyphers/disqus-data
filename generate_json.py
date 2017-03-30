import pandas as pd
import numpy as np
from collect_data import *
from analyze_data import *
from topic_modeler import *

def generate_details(data, df=None, path='www/data/forum-details.json', e=2, r=3, **kwargs):
    """
    generate json document with details for each forum

    data (DataPuller): data puller object that has all our data
    path (str): where to save the document
    """
    activity = data.get_forum_activity()
    forum_to_users = data.get_deduped_ftu()

    print 'assembling edges...'
    edges = data.get_forum_edges(dedup=True)
    print 'done'

    print 'building link matrix...'
    if df is None:
        df = build_link_matrix(data, **kwargs)
    print 'done'

    cor = get_correlations(df)

    # generate MCL groups
    groups = do_mcl(cor, e, r)
    rev_groups = {}
    for i, (k, group) in enumerate(groups.items()):
        for forum in group:
            rev_groups[forum] = k

    out = {}

    for f, details in data.forum_details.iteritems():
        act = activity.get(f)
        if f in edges:
            out_links = [u for u in forum_to_users[f] if u in data.user_to_forums]
            connectivity = sum(edges[f].values()) / float(len(out_links))
        else:
            connectivity = 0

        category = details['category']
        if category == 'null' or category is None or category == '':
            category = None

        if category is None and f.startswith('channel-'):
            category = 'Channel'

        out[f] = {
            'name': details['name'],
            'description': details['description'],
            'category': category,
            'group': rev_groups.get(f, None),
            'url': details['url'],
            'alexa': details.get('alexaRank', 0),
            'activity': act,
        }

    with open(path, 'w') as f:
        json.dump(out, f)


def generate_topics(data, model=None, path='www/data/forum-topics.json', min_docs=5):
    if model is None:
        model = TopicModeler(data)
        model.train(sample_size=2500)

    # only send data on forums with enough documents
    forums = [f for f, docs in model.docs.items() if len(docs) >= min_docs]
    topics = model.predict_topics_forums(forums)

    # send the relative incidence of each topic
    rel_topics = topics.apply(lambda row: row / model.baseline_topics, axis=1)
    rel_topics.transpose().to_json(path)


def generate_correlations(data, df=None, sortby=None,
                          path='www/data/forum-correlations.json', **kwargs):
    print 'building dataframe...'
    if df is None:
        df = build_link_matrix(data, **kwargs)

    print 'building correlation matrix...'

    # sort the forums in the matrix by some value from data.forum_details
    if sortby is not None:
        # for each forum in the index, get the det
        column = [data.forum_details[f][sortby] for f in df.index]
        key = '_' + sortby
        df[key] = pd.Categorical(column)
        df = df.sort_values(key)
        del df[key]

    cor_df = get_correlations(df)
    cor_df.to_json(path, orient='split')


def generate_cluster_graph(data, df=None, path='www/data/d3-forums.json',
                           categories=False, e=2, r=3, **kwargs):
    """
    generate a d3-parseable graph representation of the correlations between
    forums.

    df (pd.DataFrame): if provided, don't build a new link matrix
    path (str): where to save the document
    categories (bool): if true, group forums by official category. Otherwise,
        group using MCL
    e (int): passed on to MCL
    r (float): passed on to MCL
    **kwargs: passed on to build_link_matrix
    """

    if df is None:
        df = build_link_matrix(data, **kwargs)
    cor = get_correlations(df)
    nodes = []
    links = []

    # set up mapping from forum to group
    if categories:
        rev_groups = {f: data.forum_details[f]['category'] for f in cor.index}
    else:
        groups = do_mcl(cor, e, r)
        rev_groups = {}
        for i, (k, group) in enumerate(groups.items()):
            for forum in group:
                rev_groups[forum] = k

    # create node json for each forum
    for f in cor.index:
        weights = data.get_forum_activity()
        for k in weights:
            weights[k] = max(np.log(weights[k] / float(10000)), 1) * 5

        nodes.append({'id': f,
                      'group': rev_groups[f],
                      'name': data.forum_details[f]['name'],
                      'radius': weights[f]})

    # now, the tricky part: which links to include?
    for i, f1 in enumerate(cor.index):
        f1_top_5 = sorted(cor[f1])[-5]

        # iterate over all forums up to and excluding this one
        for f2 in cor.columns[:i]:
            f2_top_5 = sorted(cor[f2])[-5]

            # cull weak links
            # cor[f2][f1] > 0.9 or
            if cor[f2][f1] > 0.25 and (cor[f2][f1] >= f1_top_5 or
                                      cor[f2][f1] >= f2_top_5):
                # ordering doesn't really matter here, the matrix is symmetrical
                links.append({'source': f1, 'target': f2, 'value': cor[f2][f1]})

    out = {'nodes': nodes, 'links': links}
    with open(path, 'w') as f:
        json.dump(out, f)
