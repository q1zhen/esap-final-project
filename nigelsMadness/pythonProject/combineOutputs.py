import json
import os
import geopandas as gpd
import networkx as nx

# Load the files from output and puththem into /merge
cleaned = {}
uncleaned = {}
cities = {}
for f in os.listdir('output'):
    if not os.path.isdir(f):
        continue
    cleaned[f] = json.load(open(f'output/{f}/cleaned_{f}.json'))
    uncleaned[f] = json.load(open(f'output/{f}/uncleaned_{f}.json'))
    cities[f] = json.load(open(f'output/{f}/cities_{f}.json'))

# Create a new graph
GClean = nx.Graph()
for region in cleaned:
    GClean.add_nodes_from(cleaned[region])
    for node in cleaned[region]:
        if node not in GClean.nodes():
            GClean.add_node(node)

# other graphs too
GUnclean = nx.Graph()
for region in uncleaned:
    GUnclean.add_nodes_from(uncleaned[region])
    for node in uncleaned[region]:
        if node not in GUnclean.nodes():
            GUnclean.add_node(node)

GCities = nx.Graph()
for region in cities:
    GCities.add_nodes_from(cities[region])
    for node in cities[region]:
        if node not in GCities.nodes():
            GCities.add_node(node)

# Save the graphs
os.makedirs('output/merged', exist_ok=True)
with open('output/merged/cleaned.json', 'w+') as f:
    json.dump(list(GClean.nodes()), f)
with open('output/merged/uncleaned.json', 'w+') as f:
    json.dump(list(GUnclean.nodes()), f)
with open('output/merged/cities.json', 'w+') as f:
    json.dump(list(GCities.nodes()), f)




