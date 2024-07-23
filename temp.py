import json

with open('parsed.json', 'r') as file:
	data = json.load(file)

def point(p):
	return str([int(p[0] / 20000), int(p[1] / 20000)])

points = set()
for d in data:
	points.add(point(d["start"]))
	points.add(point(d["end"]))

idToCoord = {}
neighbors = {}

for d in data:
	idToCoord[d["id"]] = [d["start"], d["end"]]
	neighbors[d["id"]] = []

for d in data:
	print("\r", d["id"],"/",data[-1]["id"], end="")
	for d2 in data:
		if d["id"] != d2["id"]:
			if d["end"] == d2["start"]:
				neighbors[d["id"]].append(d2["id"])
			if d["start"] == d2["end"]:
				neighbors[d2["id"]].append(d["id"])

print(neighbors)
