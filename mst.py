import json
import matplotlib.pyplot as plt
import random
import sys

COLORS = ["#10ac84", "#0abde3", "#ee5253", "#ff9f43"]

def randomColor():
	return random.choice(COLORS)

location = "World"
class Edge():
	def __init__(s, w, n1, n2):
		s.weight = w
		s.node1 = n1
		s.node2 = n2

with open('centers_WORLD_k50.json', 'r') as file:
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

IS_MINIMETRO = len(nodes) < 100
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

def miniMetro(start, end):
	x = end[0] - start[0]
	y = end[1] - start[1]
	if y < 0:
		return miniMetro(end, start)
	else:
		if random.randint(0, 1):
			if abs(x) > y:
				return [start[0], end[0] - x / abs(x) * y, end[0]], [start[1], start[1], end[1]]
			else:
				return [start[0], start[0], end[0]], [start[1], end[1] - abs(x), end[1]]
		else:
			if abs(x) > y:
				return [start[0], start[0] + x / abs(x) * y, end[0]], [start[1], end[1], end[1]]
			else:
				return [start[0], end[0], end[0]], [start[1], start[1] + abs(x), end[1]]

# plt.axes().set_aspect('equal')
# plt.figure(figsize=(8, 8))
# for item in result:
# 	start = id_map[item.node1]
# 	end = id_map[item.node2]
# 	plt.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=2)


# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title(f'Track Lines of {location.capitalize()}')

# print("Plotting.")
# plt.savefig(f'plots/mst_{location}_{len(nodes)}.png', dpi=2000)
# plt.show()

# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title(f'Track Lines of {location.capitalize()} [metro style]')
# Set the figure and axes background color to black

fig, ax = plt.subplots(facecolor='black')
ax.set_facecolor('black')
ax.set_aspect('equal')
for item in result:
    start = id_map[item.node1]
    end = id_map[item.node2]
    x, y = miniMetro(start, end)
    if IS_MINIMETRO:
        plt.plot(x, y, color=randomColor(), linewidth=5)
        plt.scatter([start[0], end[0]], [start[1], end[1]],
                    color="black", edgecolor="white", linewidths=2, s=140, zorder=5)
    else:
        plt.plot(x, y, color=randomColor(), linewidth=.7)
        plt.scatter([start[0], end[0]], [start[1], end[1]],
                    color="black", edgecolor="white", linewidths=0.1, s=1, zorder=5)

# Remove ticks
plt.xticks([])
plt.yticks([])

# Save the plot with a black background
plt.savefig(f'plots/mst_{location}_{len(nodes)}_minimetrov4.png', dpi=2000, facecolor='black')
plt.show()

