import numpy as np


def audio_spectrum_spread(frequencies, centroid, weights):
    numerator = 0
    denominator = 0

    for i in range(len(frequencies)):
        fi = frequencies[i]
        Pi = weights[i]

        numerator += (fi - centroid) ** 2 * Pi
        denominator += Pi

    ASS = np.sqrt(numerator / denominator)

    return ASS



frequencies = np.array([100, 200, 300, 400, 500])
centroid = 300
weights = np.array([0.2, 0.3, 0.1, 0.4, 0.5])

result = audio_spectrum_spread(frequencies, centroid, weights)

print("Audio Spectrum Spread (ASS):", result)
