from lxml import html
import requests
import csv
import pdb
from orm import *

url_base = 'http://www.allsides.com/bias/bias-ratings?' +\
    'field_news_source_type_tid=%d' +\
    '&field_news_bias_nid=1&field_featured_bias_rating_value=All&title=&page=1'

name_elt = 'td[@class="views-field views-field-title source-title"]/a/text()'
url_elt = 'td[@class="views-field views-field-title source-title"]/a/@href'
bias_elt = 'td[@class="views-field views-field-field-bias-image"]/a/@href'
type_dict = {'Author': 1, 'News Media': 2, 'Think Tank / Policy Group': 3}
bias_dict = {'allsides': 0, 'left': 1, 'left-center': 2,
             'center': 3, 'right-center': 4, 'right': 5}


def scrape_allsides():
    _, session = get_mysql_session(remote=False)

    for category, type_id in list(type_dict.items()):
        outrows = []
        page = requests.get(url_base % type_id)
        tree = html.fromstring(page.content)

        # the first row is the header -- don't want that
        tableRows = tree.xpath('//tr')[1:]

        for tableRow in tableRows:
            source_name = ''.join(tableRow.xpath(name_elt))
            if not source_name:
                pdb.set_trace()

            print('found source', source_name)

            if '(cartoonist)' in source_name:
                source_name = source_name[0:len(source_name) - 13]
                source_category = 'Cartoonist'
            else:
                source_category = category

            # TODO: is this necessary?
            source_url = 'http://www.allsides.com' +\
                ''.join(tableRow.xpath(url_elt))

            # this td element has a nested <a href> pointing to
            # /bias/<bias-rating>
            bias_raw = ''.join(tableRow.xpath(bias_elt))[6:]
            bias = bias_dict[bias_raw]

            # try to find a forum in the database
            forum = session.query(Forum).filter(
                Forum.name.ilike('%{0}%'.format(source_name))).first()

            if forum is not None:
                print("Matched %s with %s" % (source_name, forum.name))
                forum_pk = forum.pk
            else:
                print("Found no match for", source_name)
                forum_pk = None

            entry = AllSidesEntry(name=source_name, forum_pk=forum_pk,
                                  bias=bias, category=source_category)
            session.add(entry)
            session.commit()

if __name__ == '__main__':
    scrape_allsides()
