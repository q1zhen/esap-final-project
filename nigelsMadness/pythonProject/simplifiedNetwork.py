# import sys
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
    # return G

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


def find_concentrated_points(G, k, pos=None):
    """
    Finds large concentrated points (cities) in a NetworkX graph using k-means clustering.

    Parameters:
    - G: networkx.Graph - The input graph.
    - k: int - The number of clusters to find.
    - pos: dict (optional) - A dictionary with nodes as keys and positions as values. If not provided, a spring layout will be used.

    Returns:
    - clusters: dict - A dictionary with cluster labels as keys and lists of nodes as values.
    - centroids: dict - A dictionary with cluster labels as keys and centroid positions as values.
    """

    # If no positions are provided, use a spring layout
    if pos is None:
        pos = nx.spring_layout(G)

    # Extract positions into a numpy array
    nodes = list(G.nodes)
    positions = np.array([pos[node] for node in nodes])

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(positions)

    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Organize nodes by their cluster labels
    clusters = {i: [] for i in range(k)}
    for node, label in zip(nodes, labels):
        clusters[label].append(node)

    # Convert centroids to a dictionary with cluster labels as keys
    centroid_dict = {i: centroids[i] for i in range(k)}

    return clusters, centroid_dict

# Load the shapefile
shapefile_path = 'australia-latest-free.shp/gis_osm_railways_free_1.dbf'
gdf = gpd.read_file(shapefile_path)

# Create a graph
G = nx.Graph()

# Add edges to the graph, excluding self-loops
for index, row in gdf.iterrows():
    start_point = (row['geometry'].coords[0][0], row['geometry'].coords[0][1])
    end_point = (row['geometry'].coords[-1][0], row['geometry'].coords[-1][1])
    if start_point != end_point:  # Check to avoid self-loops
        length = row['length'] if 'length' in gdf.columns else 0  # Default to 0 if no length column
        # print(dict(row))
        G.add_edge(start_point, end_point, length=length, data=row)

simplified = removeBridgeNodes(G)
# Plot the network
pos = {node: (node[0], node[1]) for node in G.nodes()}
nx.draw(G, pos, node_size=2, edge_color='green', node_color='#F8991720', width=3, with_labels=False)
nx.draw(simplified, pos, node_size=1, edge_color='red', node_color='#99999920', width=2, with_labels=False)
# draw cities
cities = getCities(simplified)
kmeanCities, centroids = find_concentrated_points(simplified, 10, pos)
print(centroids)
pos = {tuple(node): (node[0], node[1]) for node in map(tuple, centroids.values())}
# Convert centroids.values() to a list of tuples
nodelist = [tuple(node) for node in centroids.values()]

# Use the modified nodelist in the draw_networkx_nodes function
nx.draw_networkx_nodes(simplified, pos, nodelist=nodelist, node_size=30, node_color='#0000FF90')
plt.title('Railroad Network')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('simplified_graph_high_res.png', dpi=200)
plt.show()

# save the simplified graph
with open('simplified_graph.pkl', 'wb+') as f:
    pickle.dump(simplified, f)
