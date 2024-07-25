import json
import os
import sys

import geopandas as gpd
import networkx as nx

# Load the files from output and put them into /merge
cleaned = {}
uncleaned = {}
cities = {}
cleanedEdges = {}
uncleanedEdges = {}
for f in os.listdir('output'):
    if ".DS_Store" in f or 'merged' in f:
        continue
    try:
        cleaned[f] = json.load(open(f'output/{f}/cleaned_nodes_{f}.json'))
        uncleaned[f] = json.load(open(f'output/{f}/uncleaned_nodes_{f}.json'))
        cities[f] = json.load(open(f'output/{f}/cities_{f}.json'))
        cleanedEdges[f] = json.load(open(f'output/{f}/cleaned_edges_{f}.json'))
        uncleanedEdges[f] = json.load(open(f'output/{f}/uncleaned_edges_{f}.json'))
    except:
        try:
            cleaned[f] = json.load(open(f'output/{f}/cleaned_{f}.json'))
            uncleaned[f] = json.load(open(f'output/{f}/uncleaned_{f}.json'))
            cities[f] = json.load(open(f'output/{f}/cities_{f}.json'))
            cleanedEdges[f] = []
            uncleanedEdges[f] = []
            print(f"Warn: {f}")
        except:
            print(f"Failed to load {f}")
            sys.exit(1)
def getTupledBozo(nodes, edges):
    betterEdges = []
    for edge in edges:
        betterEdges.append((tuple(edge[0]), tuple(edge[1])))
    edges = tuple(betterEdges)
    betterNodes = []
    for node in nodes:
        if isinstance(node, list):
            betterNodes.append(tuple(node))
    nodes = tuple(betterNodes)
    return nodes, edges
# Create a new graph
GClean = nx.Graph()
for region in cleaned:
    processed = getTupledBozo(cleaned[region], cleanedEdges[region])
    GClean.add_nodes_from(processed[0])
    GClean.add_edges_from(processed[1])

GUnclean = nx.Graph()
for region in uncleaned:
    processed = getTupledBozo(uncleaned[region], uncleanedEdges[region])
    GUnclean.add_nodes_from(processed[0])
    GUnclean.add_edges_from(processed[1])

GCities = nx.Graph()
for region in cities:
    nodes = cities[region]
    if nodes and isinstance(nodes[0], list):  # If nodes are lists, convert to tuples or another hashable type
        nodes = tuple(tuple(node) for node in nodes)
    GCities.add_nodes_from(nodes)

# Save the graphs
os.makedirs('output/merged', exist_ok=True)
with open('output/merged/cleaned_nodes.json', 'w+') as f:
    json.dump(list(GClean.nodes()), f)
with open('output/merged/uncleaned_nodes.json', 'w+') as f:
    json.dump(list(GUnclean.nodes()), f)
with open('output/merged/cities.json', 'w+') as f:
    json.dump(list(GCities.nodes()), f)
with open('output/merged/cleaned_edges.json', 'w+') as f:
    json.dump(list(GClean.edges()), f)
with open('output/merged/uncleaned_edges.json', 'w+') as f:
    json.dump(list(GUnclean.edges()), f)