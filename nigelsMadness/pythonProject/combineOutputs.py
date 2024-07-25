import json
import os
import geopandas as gpd
import networkx as nx

# Load the files from output and puththem into /merge
cleaned = {}
uncleaned = {}
cities = {}
for f in os.listdir('output'):
    if ".DS_Store" in f or 'merged' in f:
        continue
    cleaned[f] = json.load(open(f'output/{f}/cleaned_{f}.json'))
    uncleaned[f] = json.load(open(f'output/{f}/uncleaned_{f}.json'))
    cities[f] = json.load(open(f'output/{f}/cities_{f}.json'))

# Create a new graph
GClean = nx.Graph()
for region in cleaned:
    # Ensure nodes are hashable; assuming cleaned[region] is a list of nodes
    nodes = (cleaned[region])
    if nodes and isinstance(nodes[0], list):  # If nodes are lists, convert to tuples or another hashable type
        nodes = [tuple(node) for node in nodes]
    elif nodes and isinstance(nodes[0], float):  # If nodes are floats, convert to strings or another suitable format
        nodes = [str(node) for node in nodes]
    GClean.add_nodes_from(nodes)

GUnclean = nx.Graph()
for region in uncleaned:
    # Ensure nodes are hashable; assuming cleaned[region] is a list of nodes
    nodes = uncleaned[region]
    if nodes and isinstance(nodes[0], list):  # If nodes are lists, convert to tuples or another hashable type
        nodes = [tuple(node) for node in nodes]
    elif nodes and isinstance(nodes[0], float):  # If nodes are floats, convert to strings or another suitable format
        nodes = [str(node) for node in nodes]
    GUnclean.add_nodes_from(nodes)

GCities = nx.Graph()
for region in cities:
    # Ensure nodes are hashable; assuming cleaned[region] is a list of nodes
    nodes = cities[region]
    if nodes and isinstance(nodes[0], list):  # If nodes are lists, convert to tuples or another hashable type
        nodes = [tuple(node) for node in nodes]
    elif nodes and isinstance(nodes[0], float):  # If nodes are floats, convert to strings or another suitable format
        nodes = [str(node) for node in nodes]
    GCities.add_nodes_from(nodes)

# Save the graphs
os.makedirs('output/merged', exist_ok=True)
with open('output/merged/cleaned.json', 'w+') as f:
    json.dump(list(GClean.nodes()), f)
with open('output/merged/uncleaned.json', 'w+') as f:
    json.dump(list(GUnclean.nodes()), f)
with open('output/merged/cities.json', 'w+') as f:
    json.dump(list(GCities.nodes()), f)




