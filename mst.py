import json
import matplotlib.pyplot as plt

class Edge():
	def __init__(s, w, n1, n2):
		s.weight = w
		s.node1 = n1
		s.node2 = n2

with open('centers.json', 'r') as file:
	raw = json.load(file)
# with open('nigelsMadness\pythonProject\output\england\cleaned_england.json', 'r') as file:
# 	raw = json.load(file)

with open('norm_centrality.json', 'r') as file:
	cen = json.load(file)

print("File loaded.")

nodes = set()
def coord(c):
	return (c[0], c[1])

def distance(id1, id2):
	c1 = id_map[id1]
	c2 = id_map[id2]
	return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** 0.5

# for i in raw:
# 	nodes.add(coord(i["start"]))
# 	nodes.add(coord(i["end"]))

for i in raw: nodes.add(coord(i))

id_map = {idx: node for idx, node in enumerate(nodes)}

print("Node IDs mapped.")


edges: list[Edge] = []
for id in range(len(id_map)):
	p = int(id / len(id_map) * 100)
	print(p, "%\t", "#" * p, sep="", end="\r", flush=True)
	for jd in range(id + 1, len(id_map)):
		edges.append(Edge(distance(id, jd), id, jd))

print("Edges calculated.", flush=True)

class UnionFind:
	def __init__(self, size): # just turn it off man
		self.parent = list(range(size))
		self.rank = [0] * size

	def find(self, u):
		if self.parent[u] != u:
			self.parent[u] = self.find(self.parent[u])
		return self.parent[u]

	def union(self, u, v):
		root_u = self.find(u)
		root_v = self.find(v)
		if root_u != root_v:
			if self.rank[root_u] > self.rank[root_v]:
				self.parent[root_v] = root_u
			elif self.rank[root_u] < self.rank[root_v]:
				self.parent[root_u] = root_v
			else:
				self.parent[root_v] = root_u
				self.rank[root_u] += 1
			return True
		return False

def kruskal(edges):
	node_ids = set()
	for edge in edges:
		node_ids.add(edge.node1)
		node_ids.add(edge.node2)
	id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
	n = len(node_ids)
	edges.sort(key=lambda e: e.weight)
	uf = UnionFind(n)
	result = []
	for edge in edges:
		u_idx = id_map[edge.node1]
		v_idx = id_map[edge.node2]
		if uf.union(u_idx, v_idx):
			result.append(edge)
	return result

result = kruskal(edges)

print("Result generated.")

plt.figure(figsize=(8, 8))
for item in result:
	start = id_map[item.node1]
	end = id_map[item.node2]
	plt.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=0.5)

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Track Lines')

print("Plotting.")
plt.show()

