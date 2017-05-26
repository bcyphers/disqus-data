import pandas as pd
import numpy as np
from collect_data import *
from analyze_data import *
from topic_modeler import *

def clean_cat(category, name):
    if category == 'null' or category is None or category == '':
        category = None

    if category is None and name.startswith('channel-'):
        category = 'Channel'

    return category


def generate_details(data, df=None, path='www/data/details.json', e=2, r=3, **kwargs):
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

    print 'doing MCL...'
    cor = get_correlations(df)

    # generate MCL groups
    groups = do_mcl(cor, e, r)
    rev_groups = {}
    for i, (k, group) in enumerate(groups.items()):
        for forum in group:
            rev_groups[forum] = k
    print 'done'

    out = {}

    print 'dumping json...'
    forums = [i for i in data.forum_details.iteritems() if i[0] in df.index]
    for f, details in forums:
        act = activity.get(f)
        if f in edges:
            out_links = [u for u in forum_to_users[f] if u in data.user_to_forums]
            connectivity = sum(edges[f].values()) / float(len(out_links))
        else:
            connectivity = 0

        category = clean_cat(details['category'], f)

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

    print 'done'

    return df


def generate_topics(data, model=None, path='www/data/topics.json', min_docs=5):
    if model is None:
        model = TopicModeler(data)
        model.train(sample_size=2500)

    # only send data on forums with enough documents
    doc_counts = {f: len(docs) for f, docs in model.docs.items() if len(docs) >= min_docs}
    topics = model.predict_topics_forums(doc_counts.keys())
    topics.ix['_baseline'] = model.baseline_topics

    # send the relative incidence of each topic
    topics.transpose().to_json(path)

    return model


def generate_correlations(data, df=None, path='www/data/correlations.json', **kwargs):
    print 'building dataframe...'
    if df is None:
        df = build_link_matrix(data, **kwargs)

    print 'building correlation matrix...'

    cor_df = get_correlations(df)
    cor_df.to_json(path, orient='split')

    return df


def generate_corr_scatter(data, df=None, sortby=None,
                          path='www/data/corr-scatter.json', **kwargs):
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
        df.sort_values(key, inplace=True)
        del df[key]

    cor_df = get_correlations(df)
    cor_data = json.loads(cor_df.to_json(orient='split'))
    min_max = {f: [min(df.ix[f]), max(df.ix[f])] for f in df.index}
    point_data = []
    for c in df.columns:
        points = {f: df.ix[f, c] for f in df.index}
        points['group'] = clean_cat(data.forum_details[c]['category'], c)
        points['id'] = c
        point_data.append(points)

    out_data = {
        'var': list(cor_df.index),
        'corr': cor_data['data'],
        'minMax': min_max,
        'points': point_data,
    }
    json.dump(out_data, open(path, 'w'))

    return df


def generate_cluster_graph(data, df=None, cor_cutoff=0.5,
                           path='www/data/force-graph.json', **kwargs):
    """
    generate a d3-parseable graph representation of the correlations between
    forums.

    df (pd.DataFrame): if provided, don't build a new link matrix
    cor_cutoff (float): only correlations at least this strong will be included
        as links
    path (str): where to save the document
    **kwargs: passed on to build_link_matrix
    """
    if df is None:
        print "building link matrix..."
        df = build_link_matrix(data, **kwargs)
        print "done"

    cor = get_correlations(df)
    all_cor = []
    for i in range(len(cor.index)):
        for j in range(i):
            all_cor.append(cor.ix[i, j])
    all_cor.sort(reverse=True)

    nodes = []
    links = []

    print "building nodes..."
    # create node json for each forum
    for f in cor.index:
        weights = data.get_forum_activity()
        for k in weights:
            weights[k] = max(np.log(weights[k] / float(10000)), 1) * 5

        nodes.append({'id': f,
                      'name': data.forum_details[f]['name'],
                      'radius': weights[f]})
    print "done"

    print "building links..."
    # now, the tricky part: create the links
    # start by iterating over all forums
    for i, f1 in enumerate(cor.index):
        # get the value of the fifth highest correlation with this forum
        f1_top_5 = sorted(cor[f1])[-5]

        # iterate over all forums up to and excluding this one
        for f2 in cor.columns[:i]:
            f2_top_5 = sorted(cor[f2])[-5]

            # only include links of sufficient strength, or links in the top 5
            #if cor[f2][f1] > all_cor[len(nodes) * 3]:
            if cor[f2][f1] > cor_cutoff and (cor[f2][f1] >= f1_top_5 or
                                             cor[f2][f1] >= f2_top_5):
                # ordering doesn't really matter here, the matrix is symmetrical
                links.append({'source': f1, 'target': f2, 'value': cor[f2][f1]})
    print "done"

    print "dumping json..."
    out = {'nodes': nodes, 'links': links}
    with open(path, 'w') as f:
        json.dump(out, f)

    return df
