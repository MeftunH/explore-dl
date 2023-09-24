import glob
import time
import matplotlib.pyplot as plt
import librosa.display

### https://www.openslr.org/12 you can download from there
folder_path = '/dev-clean/1462/'

for filename in glob.glob(f'{folder_path}/**/*.wav', recursive=True):
    speech_path = filename
    x, sr = librosa.load(speech_path)

    plt.figure()
    librosa.display.waveshow(x, sr=sr)
    time.sleep(0.1)
    plt.pause(0.0001)

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    print(Xdb)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

plt.show()
