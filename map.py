import json

with open('merged.json', 'r') as file:
	raw = json.load(file)

def coord(c):
	return (c[0], c[1])

nodes = set()
coord2id = {}
id2coord = {}

edges = set()

for edge in raw:
	start = coord(edge["start"])
	end = coord(edge["end"])
	for node in [start, end]:
		if node in nodes: 
			continue
		else:
			coord2id[node] = len(nodes)
			id2coord[len(nodes)] = node
			nodes.add(node)
	edges.add((coord2id[start], coord2id[end]))

with open('graph.json', 'w+') as f:
	f.write(json.dumps({"nodes": id2coord, "edges": list(edges)}))
