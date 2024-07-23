import json
import matplotlib.pyplot as plt

# Load the JSON data
with open('parsed.json', 'r') as file:
	data = json.load(file)

# Create a plot
plt.figure(figsize=(10, 10))

# Plot each line from start to end
for item in data:
	start = item['start']
	end = item['end']
	plt.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=0.5)

# Add labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Track Lines')

# Show the plot
plt.show()
