import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import json
import os

for location in ["vietnam"]:
	with open(f'nigelsMadness/pythonProject/output/{location}-latest-free.shp/cleaned_edges_{location}-latest-free.shp.json', 'r') as file:
		raw = json.load(file)
	plt.axes().set_aspect('equal')
	for edge in raw:
		start = edge[0]
		end = edge[1]
		plt.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=2)
	plt.title(f'Real Track Lines of {location.capitalize()}')
	plt.xlabel('X Coordinate')
	plt.ylabel('Y Coordinate')
	plt.savefig(f'cheezhenPlots/real_{location}_cleaned.png', dpi=400)
	plt.show()