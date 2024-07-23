import json

with open('parsed.json', 'r') as file:
	data = json.load(file)

def point(p):
	return str([int(p[0] / 20000), int(p[1] / 20000)])

points = set()
for d in data:
	points.add(point(d["start"]))
	points.add(point(d["end"]))

print(len(points))