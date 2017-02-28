import pandas as pd
import numpy as np
from collect_data import *
from analyze_data import *

def generate_details(data, df=None, path='www/forum-details.json'):
    """
    generate json document with details for each forum

    df (pd.DataFrame): if provided, don't build a new link matrix
    path (str): where to save the document
    """
    if df is None:
        print 'building link matrix...'
        df = build_link_matrix(data)
        print 'done'

    activity = data.get_forum_activity()
    forum_to_users = data.get_deduped_ftu()

    print 'assembling edges...'
    edges = data.get_forum_edges(dedup=True)
    print 'done'

    out = {}

    for f, details in data.forum_details.iteritems():
        act = activity.get(f)
        if f in edges:
            out_links = [u for u in forum_to_users[f] if u in data.user_to_forums]
            connectivity = sum(edges[f].values()) / float(len(out_links))
        else:
            connectivity = 0

        out[f] = {
            'name': details['name'],
            'description': details['description'],
            'category': details['category'],
            'url': details['url'],
            'connectivity': connectivity,
            'activity': act,
        }

    with open(path, 'w') as f:
        json.dump(out, f)


def generate_topics(data, path='www/forum-details.json'):
    pass


def generate_correlations(data, path='www/forum-details.json'):
    pass


def generate_cluster_graph(data, df=None, path='www/d3-forums.json', categories=True,
                           e=2, r=3, **kwargs):
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

    # set up links between each pair of forums
    for i, f1 in enumerate(cor.index):
        weights = data.get_forum_activity()
        for k in weights:
            weights[k] = max(np.log(weights[k] / float(10000)), 1) * 5

        node = {'id': f1,
                'group': rev_groups[f1],
                'name': data.forum_details[f1]['name'],
                'radius': weights[f1]}
        nodes.append(node)

        f1_top_5 = sorted(cor[f1])[-5]

        for f2 in cor.columns[:i]:
            f2_top_5 = sorted(cor[f2])[-5]
            # cull weak links
            if cor[f2][f1] > 0.1 and (cor[f2][f1] >= f1_top_5 or
                                      cor[f2][f1] >= f2_top_5):
                # ordering doesn't really matter here, the matrix is symmetrical
                link = {'source': f1, 'target': f2, 'value': cor[f2][f1]}
                links.append(link)

    out = {'nodes': nodes, 'links': links}
    with open(path, 'w') as f:
        json.dump(out, f)
