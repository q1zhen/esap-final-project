import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import json
import os

location = "vietnam"
with open(f'nigelsMadness/pythonProject/output/{location}-latest-free.shp/cleaned_nodes_{location}-latest-free.shp.json', 'r') as file:
	raw = json.load(file)

coordinates = np.array(raw)

k_max = 30
k_range = range(2, k_max)

silhouette_scores = []

for k in k_range:
	kmeans = KMeans(n_clusters=k, random_state=0).fit(coordinates)
	labels = kmeans.labels_
	score = silhouette_score(coordinates, labels)
	silhouette_scores.append(score)
	p = int((k / k_max) * 100)
	print("[", "#" * p, " " * (100 - p), "] ", p, "%", sep="", end="\r", flush=True)
print("Finished.", flush=True)

optimal_k = k_range[np.argmax(silhouette_scores)]

print(f"The optimal number of clusters (k) is: {optimal_k}")

plt.plot(k_range, silhouette_scores, marker='o')
plt.title(f'Silhouette Scores for Different k Values [{location.capitalize()}]')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.savefig(f'cheezhenPlots/kmeans_{location}_silhouette_.png', dpi=400)
plt.show()

# Not fun stuff here
# kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(coordinates)
# labels = kmeans.labels_
# centers = kmeans.cluster_centers_
#
# plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap='viridis')
# plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
# plt.title(f'K-Means Clustering with k={optimal_k}')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.show()

# fun stuff here

l = input("Enter k values: ")
kl = [int(i) for i in l.split(",")]

for k in kl:
	plt.axes().set_aspect('equal')
	print(k)
	kmeans = KMeans(n_clusters=k, random_state=0).fit(coordinates)

	labels = kmeans.labels_

	centers = kmeans.cluster_centers_

	with open('centers.json', 'w+') as file:
		file.write(json.dumps([
			list(i) for i in centers
		]))

	plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap='viridis', s=10)
	plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
	plt.title(f'K-Means Clustering (k = {k}) [{location.capitalize()}]')
	plt.xlabel('X Coordinate')
	plt.ylabel('Y Coordinate')
	plt.savefig(f'cheezhenPlots/kmeans_{location}_{k}_.png', dpi=400)
	plt.show()
	os.system(f"python mst.py {location}")
