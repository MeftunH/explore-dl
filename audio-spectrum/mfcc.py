import librosa
import numpy as np
import os

print("Current working directory:", os.getcwd())

def extract_mfcc(audio_file_path, n_mfcc=13):
    """
    Extract MFCC features from an audio file.

    Parameters:
        audio_file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCCs to return.

    Returns:
        np.ndarray: MFCC features.
    """
    # Load audio file
    y, sr = librosa.load(audio_file_path)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return mfccs


# Example usage
audio_file_path = "/dev-clean/174/50561/174-50561-0002.wav"
mfcc_features = extract_mfcc(audio_file_path)

print(f"MFCC features shape: {mfcc_features.shape}")
