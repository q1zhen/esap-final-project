# import sys
import json
import os

import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.cluster import KMeans

# IF a node only serves to bridge two edges, remove it and connect the two edges directly with a new single edge
def removeBridgeNodes(G):
    outputGraph = G.copy()
    bridgeNodesList = []
    for node in G.nodes():
        if len(list(G.neighbors(node))) == 0:
            if node in outputGraph.nodes():
                outputGraph.remove_node(node)
        if len(list(G.neighbors(node))) == 1:
            # if node 1 only connects to 2 but node 2 only connects back to 1, remove both
            if len(list(G.neighbors(list(G.neighbors(node))[0]))) == 1:
                if node in outputGraph.nodes():
                    outputGraph.remove_node(node)
                if list(G.neighbors(node))[0] in outputGraph.nodes():
                    outputGraph.remove_node(list(G.neighbors(node))[0])
        if len(list(G.neighbors(node))) == 2:
            bridgeNodesList.append(node)
    for node in bridgeNodesList:
        removeBridgeNode(outputGraph, node)
    return outputGraph

def removeBridgeNode(G, node):
    if node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 0:
            G.remove_node(node)
        if len(neighbors) == 2:
            edge1 = G.get_edge_data(node, neighbors[0])
            edge2 = G.get_edge_data(node, neighbors[1])
            G.remove_node(node)
            G.add_edge(neighbors[0], neighbors[1], length=edge1['length'] + edge2['length'])
        else:
            pass
    return G

def getCities(G):
    radius = 0.5
    cities = {}
    for node in G.nodes():
        matched = False
        for city in cities.keys():
            if abs(node[0]-city[0]) < radius and abs(node[1]-city[1]) < radius:
                # break out and continue the outer loop
                matched = True
                break
        if matched:
            continue
        for node2 in G.nodes():
            if abs(node[0]-node2[0]) < radius and abs(node[1]-node2[1]) < radius:
                cities[node] = {}
                break
    return cities


def find_concentrated_points(G, max_clusters, pos=None):
    if pos is None:
        pos = nx.spring_layout(G)
    nodes = list(G.nodes)
    if not nodes:  # Check if the list of nodes is empty
        return {}, {}
    positions = np.array([pos[node] for node in nodes])
    if positions.size == 0:  # Check if positions is empty
        return {}, {}
    if len(positions.shape) == 1:  # Ensure positions is a 2D array
        positions = positions.reshape(-1, 1)

    # Dynamically adjust the number of clusters based on the number of samples
    n_clusters = min(len(positions), max_clusters)

    # If there are not enough samples to form even a single cluster, return empty results
    if n_clusters < 1:
        return {}, {}

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(positions)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    clusters = {i: [] for i in range(n_clusters)}
    for node, label in zip(nodes, labels):
        clusters[label].append(node)
    centroid_dict = {i: centroids[i] for i in range(n_clusters)}
    return clusters, centroid_dict

def snapToNetwork(coords, listOfPoints):
    currentBest = [listOfPoints[0], ((coords[0] - listOfPoints[0][0]) ** 2 + (coords[1] - listOfPoints[0][1]) ** 2) ** 0.5]
    for point in listOfPoints:
        distance = ((coords[0] - point[0]) ** 2 + (coords[1] - point[1]) ** 2) ** 0.5
        if distance < currentBest[1]:
            currentBest = (point, distance)
    return currentBest[0]

def plotWorld(file):
    # list of tuples
    nodes = [tuple(node) for node in json.load(open(file))]
    # Create a graph
    G = nx.Graph()

    # Add edges to the graph, excluding self-loops
    G.add_nodes_from(nodes)

    simplified = removeBridgeNodes(G)
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    nx.draw(G, pos, node_size=3, edge_color='green', node_color='#F8991718', width=3, with_labels=False)
    nx.draw(simplified, pos, node_size=2, edge_color='red', node_color='#00991780', width=2, with_labels=False)
    kmeanCities, centroids = find_concentrated_points(simplified, 10, pos)
    pos = {tuple(node): (node[0], node[1]) for node in map(tuple, centroids.values())}
    nodelist = [tuple(node) for node in centroids.values()]
    cities = []
    for node in nodelist:
        cities.append(snapToNetwork(node, list(simplified.nodes)))
    Gnet = nx.Graph()
    Gnet.add_nodes_from(cities)
    pos = {node: (node[0], node[1]) for node in Gnet.nodes()}
    nx.draw_networkx_nodes(Gnet, pos, node_size=30, node_color='blue')
    # create folder for region name
    regionName = "merged"
    os.makedirs(f'output/{regionName}', exist_ok=True)
    plt.savefig(f'output/{regionName}/simplified_graph_high_res.png', dpi=2000)
    plt.show()

plotWorld("output/merged/uncleaned.json")

