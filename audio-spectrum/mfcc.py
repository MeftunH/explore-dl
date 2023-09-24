import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display

audio_path = 'dev-clean/777/126732/777-126732-0002.flac'

try:
    x , sr = librosa.load(audio_path)
except Exception as e:
    print(f"An error occurred while loading the audio file: {e}")
    exit()

if x is None or sr is None:
    print("Failed to load audio.")
    exit()

print(f"Type of x: {type(x)}, Type of sr: {type(sr)}")

# Display audio
audio = ipd.Audio(audio_path)

# Plot waveform
plt.figure(figsize=(14, 5))
try:
    librosa.display.waveshow(x, sr=sr)
except Exception as e:
    print(f"An error occurred while plotting the waveform: {e}")

plt.show()

# Plot spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))

try:
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
except Exception as e:
    print(f"An error occurred while plotting the spectrogram: {e}")

plt.colorbar()
plt.show()
