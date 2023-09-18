import numpy as np

frequencies = np.array([100, 200, 300, 400, 500])
weights = np.array([0.2, 0.3, 0.1, 0.4, 0.5])

centroid = np.sum(frequencies * weights) / np.sum(weights)

sorted_frequencies = np.sort(frequencies)
cumulative_weights = np.cumsum(weights)
median_index = np.argmax(cumulative_weights >= 0.5 * np.sum(weights))
median = sorted_frequencies[median_index]

print("ASC:", centroid)
