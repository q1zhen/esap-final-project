import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pyrosm.data import sources, get_data
from sklearn.cluster import KMeans
# import osmnx as ox
from pyrosm import OSM

def removeBridgeNodes(G):
    outputGraph = G.copy()
    bridgeNodesList = []
    for node in G.nodes():
        if len(list(G.neighbors(node))) == 0:
            if node in outputGraph.nodes():
                outputGraph.remove_node(node)
        if len(list(G.neighbors(node))) == 1:
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
            if abs(node[0] - city[0]) < radius and abs(node[1] - city[1]) < radius:
                matched = True
                break
        if matched:
            continue
        for node2 in G.nodes():
            if abs(node[0] - node2[0]) < radius and abs(node[1] - node2[1]) < radius:
                cities[node] = {}
                break
    return cities


def find_concentrated_points(G, k, pos=None):
    if pos is None:
        pos = nx.spring_layout(G)
    nodes = list(G.nodes)
    positions = np.array([pos[node] for node in nodes])
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(positions)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    clusters = {i: [] for i in range(k)}
    for node, label in zip(nodes, labels):
        clusters[label].append(node)
    centroid_dict = {i: centroids[i] for i in range(k)}
    return clusters, centroid_dict


def snapToNetwork(coords, listOfPoints):
    currentBest = [listOfPoints[0],
                   ((coords[0] - listOfPoints[0][0]) ** 2 + (coords[1] - listOfPoints[0][1]) ** 2) ** 0.5]
    for point in listOfPoints:
        distance = ((coords[0] - point[0]) ** 2 + (coords[1] - point[1]) ** 2) ** 0.5
        if distance < currentBest[1]:
            currentBest = (point, distance)
    return currentBest[0]


def shapeDoDad(regionName):
    custom_filter = {"railway": ["rail", "light_rail", "subway", "tram", "monorail"]}
    print(regionName)
    fp = get_data(regionName)
    G = OSM(fp).get_data_by_custom_criteria(custom_filter)
    print(i)
    print(G)
    if G is None:
        return
    simplified = removeBridgeNodes(G)

    pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
    nx.draw(G, pos, node_size=0, edge_color='black', node_color='#F8991720', width=1, with_labels=False)

    kmeanCities, centroids = find_concentrated_points(simplified, 10, pos)
    pos = {tuple(node): (node[0], node[1]) for node in map(tuple, centroids.values())}
    nodelist = [tuple(node) for node in centroids.values()]
    cities = []
    for node in nodelist:
        cities.append(snapToNetwork(node, list(simplified.nodes)))

    Gnet = nx.Graph()
    Gnet.add_nodes_from(cities)
    pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in Gnet.nodes()}
    nx.draw_networkx_nodes(Gnet, pos, node_size=30, node_color='blue')

    os.makedirs(f'output/{regionName}', exist_ok=True)
    plt.savefig(f'output/{regionName}/simplified_graph_high_res.png', dpi=2000)
    plt.show()

    with open(f"output/{regionName}/cleaned_{regionName}.json", "w+") as f:
        json.dump(list(simplified.nodes()), f)
    with open(f"output/{regionName}/uncleaned_{regionName}.json", "w+") as f:
        json.dump(list(G.nodes()), f)
    with open(f"output/{regionName}/cities_{regionName}.json", "w+") as f:
        json.dump(cities, f)
    with open(f'output/{regionName}/simplified_graph_{regionName}.pkl', 'wb+') as f:
        pickle.dump(simplified, f)


regionNames = []
for i in sources.available.keys():
    for j in sources.available[i]:
        regionNames.append(j)
for i in regionNames:
    shapeDoDad(i)
