import json
import os
import datetime

# try to load mysql connection info from a config file
try:
    with open('mysql-conf.json') as f:
        mysql_settings = json.load(f)
except IOError:
    print "Please supply valid MySQL connection details in mysql-conf.json."
    print "See example at mysql-conf.example.json."

# TODO: this is ugly
DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data/'

# the first instant of President Trump's tenure
TRUMP_START = datetime.datetime(2017, 1, 20, 17, 0, 0)

DEDUP = {
    'channel-theatlanticdiscussions': 'theatlantic',
    'theatlanticcities': 'theatlantic',
    'theatlanticwire': 'theatlantic',
    'in-focus': 'theatlantic',
    'bwbeta': 'bloomberg',
    'bloombergview': 'bloomberg',
    'pbsnewshourformsrodeo': 'pbsnewshour',
    'pj-instapundit': 'pj-media',
    'spectator-new-blogs': 'spectator-new-www',
    'spectatorwww': 'spectator-new-www',
    'theamericanspectator': 'spectator-org',
    'spectatororg': 'spectator-org',
    'channel-theavclubafterdark': 'avclub',
    'mtonews': 'mtocom',
}
