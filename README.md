# disqus-data

Pulls down data from Disqus and saves as ugly json files, then it can do some
graph analysis and NLP things as well.

Python scripts are in the main directory. 

`collect_data.py`: the DataPuller object opens up an API session and pulls down
data from Disqus's servers. Pass it an API key and a data directory and watch it
go.

`analyze_data.py`: A bunch of functions here to generate graphs, do NLP, etc.
with data from a DataPuller object.

`generate_json.py`: a bunch of functions that take DataPuller objects, call
analyze_data functions, and output interesting data as JSON files to be
interpreted by the web page.

Blog post code in www/. Adapted disqus API library in disqusapi/.
