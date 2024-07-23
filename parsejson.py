import json

with open("raw.geojson", "r") as f:
	raw = json.load(f)

parsed = []

for line in raw["features"]:
	if line["id"] != 43559:
		parsed.append({
			"id": line["id"],
			"start": line["geometry"]["coordinates"][0],
			"end": line["geometry"]["coordinates"][-1],
			"track_gauge": line["properties"]["track_gauge"],
			"length_km": line["properties"]["length_km"],
			"tracks": line["properties"]["tracks"]
		})

with open("parsed.json", "w+") as f:
	f.write(json.dumps(parsed))
