# ESAP FINAL PROJECT!!!

The file `graph.json` is the file that will finally be used. It is in the format of:
```json
{
	"nodes": {
		"0": [x, y],
		"1": [x, y],
		...
	},
	"edges": [
		[0, 1],
		[1, 2],
		...
	]
}
```
`"nodes"` describes the coordinates of the nodes, where the key is the node's id and the value is the coordinate. `"edges"` describes a set of edges consisting of the two nodes' ids.

The file `centrality.py` reads the data from `graph.json` and generates an adjacency matrix using numpy, and you can do whatever you want then.

The file `norm.py` normalizes data in `centrality.json` and writes output to `norm_centrality.json`. Feel free to change the normalization there.

The file `mst.py` plots the final graph using relavent data. If you want to change how weight is calculated, go to the definition of the `distance()` function.
