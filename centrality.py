import numpy as np
import json
import math

with open('graph.json', 'r') as file:
	raw = json.load(file)

edges = raw["edges"]

max_node = max(max(edge) for edge in edges)
adj_matrix = np.zeros((max_node + 1, max_node + 1), dtype=int)

for edge in edges:
	u, v = edge
	adj_matrix[u, v] = 1
	adj_matrix[v, u] = 1

eigenvalues, eigenvectors = np.linalg.eig(adj_matrix)
index_of_largest_eigenvalue = np.argmax(eigenvalues)
principal_eigenvector = eigenvectors[:, index_of_largest_eigenvalue]

with open('centrality.json', 'w+') as f:
	f.write(json.dumps({str(i): float(principal_eigenvector[i]) for i in range(len(principal_eigenvector))}))
