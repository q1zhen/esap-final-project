import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

# Load the shapefile
shapefile_path = 'australia/gis_osm_railways_free_1.shp'
gdf = gpd.read_file(shapefile_path)

# Create a graph
G = nx.Graph()

# Add edges to the graph, excluding self-loops
for index, row in gdf.iterrows():
    start_point = (row['geometry'].coords[0][0], row['geometry'].coords[0][1])
    end_point = (row['geometry'].coords[-1][0], row['geometry'].coords[-1][1])
    if start_point != end_point:  # Check to avoid self-loops
        length = row['length'] if 'length' in gdf.columns else 0  # Default to 0 if no length column
        G.add_edge(start_point, end_point, length=length)

mst=nx.minimum_spanning_tree(G)
# Plot the network
pos = {node: (node[0], node[1]) for node in G.nodes()}
nx.draw(G, pos, node_size=0, edge_color='green', node_color='black', width=1, with_labels=False)
nx.draw(mst, pos, node_size=0, edge_color='red', node_color='black', width=1, with_labels=False)
plt.title('Railroad Network')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Example usage: Find the shortest path between two points (if coordinates are known)
# Adjust the coordinates to actual points in your network
# path = nx.shortest_path(G, source=(start_lon, start_lat), target=(end_lon, end_lat), weight='length')
# print(path)