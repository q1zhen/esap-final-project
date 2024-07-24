import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import json

with open('graph.json', 'r') as file:
	raw = json.load(file)

coordinates = np.array(list(raw["nodes"].values()))

# k_max = 500
# k_range = range(2, k_max)

# silhouette_scores = []

# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(coordinates)
#     labels = kmeans.labels_
#     score = silhouette_score(coordinates, labels)
#     silhouette_scores.append(score)
#     p = int((k / k_max) * 100)
#     print("[", "#" * p, " " * (100 - p), "] ", p, "%", sep="", end="\r", flush=True)
# print("Finished.", flush=True)

# optimal_k = k_range[np.argmax(silhouette_scores)]

# print(f"The optimal number of clusters (k) is: {optimal_k}")

# plt.plot(k_range, silhouette_scores, marker='o')
# plt.title('Silhouette Scores for Different k Values')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.show()

# kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(coordinates)
# labels = kmeans.labels_
# centers = kmeans.cluster_centers_

# plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap='viridis')
# plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
# plt.title(f'K-Means Clustering with k={optimal_k}')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.show()

for k in [86]:

	kmeans = KMeans(n_clusters=k, random_state=0).fit(coordinates)

	labels = kmeans.labels_

	centers = kmeans.cluster_centers_

	with open('centers.json', 'w+') as file:
		file.write(json.dumps([
			list(i) for i in centers
		]))

	plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap='viridis', s=10)
	plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
	plt.title(f'K-Means Clustering (k = {k})')
	plt.xlabel('X Coordinate')
	plt.ylabel('Y Coordinate')
	plt.show()
