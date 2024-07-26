import json
import multiprocessing
import os
import threading

import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
            if abs(node[0]-city[0]) < radius and abs(node[1]-city[1]) < radius:
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
    if not nodes:
        return {}, {}
    positions = np.array([pos[node] for node in nodes])
    if positions.size == 0:
        return {}, {}
    if len(positions.shape) == 1:
        positions = positions.reshape(-1, 1)
    n_clusters = min(len(positions), max_clusters)
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

def pressNodes(graph, radius):
    nodes = list(graph.nodes)
    merged_nodes = set()
    for i, node1 in enumerate(nodes):
        if node1 in merged_nodes:
            continue
        for node2 in nodes[i + 1:]:
            if node2 in merged_nodes:
                continue
            if abs(node1[0] - node2[0]) < radius and abs(node1[1] - node2[1]) < radius:
                graph = mergeNodes(graph, node1, node2)
                merged_nodes.add(node1)
                merged_nodes.add(node2)
    return graph

def mergeNodes(graph, node1, node2):
    if node1 in graph.nodes() and node2 in graph.nodes():
        neighbors = list(graph.neighbors(node1))
        for neighbor in neighbors:
            if neighbor != node2:
                graph.add_edge(node2, neighbor, length=graph.get_edge_data(node1, neighbor)['length'])
        graph.remove_node(node1)
    return graph

def shapeDoDad(regionName, shapefile_path):
    # check if output dir exists and skit if it does
    if os.path.exists(f'/Users/massivezappy/Desktop/ESAP/FinalProject/final-project/nigelsMadness/pythonProject/output/{regionName}'):
        return
    gdf = gpd.read_file(shapefile_path)
    G = nx.Graph()
    for index, row in gdf.iterrows():
        start_point = (row['geometry'].coords[0][0], row['geometry'].coords[0][1])
        end_point = (row['geometry'].coords[-1][0], row['geometry'].coords[-1][1])
        if start_point != end_point:
            length = row['length'] if 'length' in gdf.columns else 0
            G.add_edge(start_point, end_point, length=length, data=row)
    simplified = removeBridgeNodes(pressNodes(G, 0.0001))
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    nx.draw(G, pos, node_size=3, edge_color='green', node_color='#F8991718', width=3, with_labels=False)
    nx.draw(simplified, pos, node_size=2, edge_color='red', node_color='#00991780', width=2, with_labels=False)
    kmeanCities, centroids = find_concentrated_points(simplified, 10, pos)
    pos = {tuple(node): (node[0], node[1]) for node in map(tuple, centroids.values())}
    nodelist = [tuple(node) for node in centroids.values()]
    cities = []
    for node in nodelist:
        cities.append(snapToNetwork(node, list(simplified.nodes())))
    Gnet = nx.Graph()
    Gnet.add_nodes_from(cities)
    pos = {node: (node[0], node[1]) for node in Gnet.nodes()}
    nx.draw_networkx_nodes(Gnet, pos, node_size=30, node_color='blue')
    os.makedirs(f'/Users/massivezappy/Desktop/ESAP/FinalProject/final-project/nigelsMadness/pythonProject/output/{regionName}', exist_ok=True)
    plt.savefig(f'/Users/massivezappy/Desktop/ESAP/FinalProject/final-project/nigelsMadness/pythonProject/output/{regionName}/simplified_graph_high_res.png', dpi=2000)
    plt.close()
    with open(f"/Users/massivezappy/Desktop/ESAP/FinalProject/final-project/nigelsMadness/pythonProject/output/{regionName}/cleaned_nodes_{regionName}.json", "w+") as f:
        json.dump(list(simplified.nodes()), f)
    with open(f"/Users/massivezappy/Desktop/ESAP/FinalProject/final-project/nigelsMadness/pythonProject/output/{regionName}/uncleaned_nodes_{regionName}.json", "w+") as f:
        json.dump(list(G.nodes()), f)
    with open(f"/Users/massivezappy/Desktop/ESAP/FinalProject/final-project/nigelsMadness/pythonProject/output/{regionName}/cleaned_edges_{regionName}.json", "w+") as f:
        json.dump(list(simplified.edges()), f)
    with open(f"/Users/massivezappy/Desktop/ESAP/FinalProject/final-project/nigelsMadness/pythonProject/output/{regionName}/uncleaned_edges_{regionName}.json", "w+") as f:
        json.dump(list(G.edges()), f)
    with open(f"/Users/massivezappy/Desktop/ESAP/FinalProject/final-project/nigelsMadness/pythonProject/output/{regionName}/cities_{regionName}.json", "w+") as f:
        json.dump(cities, f)
    with open(f'/Users/massivezappy/Desktop/ESAP/FinalProject/final-project/nigelsMadness/pythonProject/output/{regionName}/simplified_graph_{regionName}.pkl', 'wb+') as f:
        pickle.dump(simplified, f)
    # print("saved files")

def process_shapefile(regionName, shapefile_path):
    print(f"Processing {regionName}")
    shapeDoDad(regionName, shapefile_path)
    print(f"Finished processing {regionName}")

def process_country(continent, country):
    shapefile_path = f'/Users/massivezappy/Desktop/ESAP/FinalProject/final-project/nigelsMadness/pythonProject/input/WORLD/{continent}/{country}/gis_osm_railways_free_1.shp'
    process_shapefile(country.replace('-latest-free.shp', ''), shapefile_path)

def process_continent(continent, countries, max_threads):
    threads = []
    for country in countries:
        t = threading.Thread(target=process_country, args=(continent, country))
        threads.append(t)
        t.start()
        if len(threads) >= max_threads:
            for t in threads:
                t.join()
            threads = []
    for t in threads:
        t.join()

def main():
    base_dir = '/Users/massivezappy/Desktop/ESAP/FinalProject/final-project/nigelsMadness/pythonProject/input/WORLD'
    max_processes = 16
    max_threads = 8
    continents = [continent for continent in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, continent))]
    with multiprocessing.Pool(processes=max_processes) as pool:
        for continent in continents:
            continent_path = os.path.join(base_dir, continent)
            countries = [country for country in os.listdir(continent_path) if os.path.isdir(os.path.join(continent_path, country))]
            pool.apply_async(process_continent, args=(continent, countries, max_threads))
        pool.close()
        pool.join()
    print("All regions processed.")

if __name__ == "__main__":
    main()