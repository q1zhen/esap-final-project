import json
import matplotlib.pyplot as plt

# Load the JSON data
with open('graph.json', 'r') as file:
	data = json.load(file)

# Create a plot
plt.figure(figsize=(8, 8))

# Plot each line from start to end
for edge in data["edges"]:
	start = data["nodes"][str(edge[0])]
	end = data["nodes"][str(edge[1])]
	plt.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=0.5)

# Add labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Track Lines')

# Show the plot
plt.show()
