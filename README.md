# disqus-data

Pulls down data from Disqus and saves posts in a MySQL database. Then it can do
some graph analysis and NLP things as well.

### Requirements

Set up python and mysql
```
sudo apt update
sudo apt install python-dev python-pip python-virtualenv
sudo apt install mysql-server libmysqlclient-dev
```

Install python packages in virtualenv
```
virtualenv venv
. venv/bin/activate
pip install cython
pip install -r requirements.txt
```


### Files
Python scripts are in the main directory. 

`collect_data.py`: the DataPuller object opens up an API session and pulls down
data from Disqus's servers. Pass it an API key and a data directory and watch it
go.

`analyze_data.py`: A bunch of functions here to generate graphs, do NLP, etc.
with data from a DataPuller object.

`generate_json.py`: a bunch of functions that take DataPuller objects, call
analyze_data functions, and output interesting data as JSON files to be
interpreted by the web page.

all others: TODO: documentation

Blog post code in www/. Adapted disqus API library in disqusapi/.
