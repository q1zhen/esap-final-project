import json
import math

with open('centrality.json', 'r') as file:
	data = json.load(file)

ids = list(data.keys())
values = list(data.values())

m = min(values)
norm = {}

for i in ids:
	norm[i] = math.log10(data[i] - m + 3)

with open('norm_centrality.json', 'w+') as f:
	f.write(json.dumps(norm))

