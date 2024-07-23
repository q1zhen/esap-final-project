import json

with open("parsed.json", "r") as f:
	raw = json.load(f)

def coord(c):
    k = 10000
    return (int(c[0] / k) * k, int(c[1] / k) * k)

merged = []

for l in raw:
	line = l
	line["start"] = coord(l["start"])
	line["end"] = coord(l["end"])
	if line["start"] != line["end"]: merged.append(line)

with open("merged.json", "w+") as f:
	f.write(json.dumps(merged))
