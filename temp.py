import json
import matplotlib.pyplot as plt

with open('norm_centrality.json', 'r') as file:
	data = json.load(file)

ids = list(data.keys())
values = list(data.values())

plt.figure(figsize=(10, 5))
plt.bar(ids, values, color='blue')
plt.xlabel('ID')
plt.ylabel('Value')
plt.title('ID vs Value')
plt.grid(axis='y')
plt.show()

