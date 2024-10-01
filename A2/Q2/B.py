import os
import librosa
import numpy as np
import pandas as pd
import scipy.signal as signal

# Function to apply a bandpass filter to the audio signal
def apply_bandpass_filter(y, sr, low_freq, high_freq):
    nyquist = 0.5 * sr
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(1, [low, high], btype='band')
    y_filtered = signal.lfilter(b, a, y)
    return y_filtered

# Function to extract relevant audio features
def extract_audio_features_with_filters(directory):
    features = []
    for foldername in os.listdir(directory):
        folder_path = os.path.join(directory, foldername)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(folder_path, filename)
                    # Load the audio file
                    y, sr = librosa.load(file_path, sr=16000)
                    
                    # Apply bandpass filter (e.g., 300 Hz to 3400 Hz for human voice)
                    y_filtered = apply_bandpass_filter(y, sr, low_freq=300, high_freq=3400)
                    
                    # Extract MFCC features (using the filtered signal)
                    mfcc = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=13)
                    mfcc_mean = np.mean(mfcc, axis=1)  # Mean MFCC across time
                    
                    # Extract Chroma features
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                    chroma_mean = np.mean(chroma, axis=1)
                    
                    # Extract Spectral features
                    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                    spectral_flatness = librosa.feature.spectral_flatness(y=y)
                    
                    # Calculate the mean and standard deviation for spectral features
                    spectral_centroid_mean = np.mean(spectral_centroid)
                    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
                    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
                    spectral_flatness_mean = np.mean(spectral_flatness)
                    
                    # Append extracted features to the list
                    features.append([
                        foldername, 
                        *mfcc_mean, 
                        *chroma_mean, 
                        spectral_centroid_mean, 
                        spectral_bandwidth_mean, 
                        *spectral_contrast_mean, 
                        spectral_flatness_mean
                    ])
    
    # Create a DataFrame for the extracted features
    mfcc_columns = [f'MFCC_{i}' for i in range(1, 14)]
    chroma_columns = [f'Chroma_{i}' for i in range(1, 13)]
    spectral_columns = ['Spectral_Centroid', 'Spectral_Bandwidth', 'Spectral_Contrast1', 
                        'Spectral_Contrast2', 'Spectral_Contrast3', 'Spectral_Contrast4', 
                        'Spectral_Contrast5', 'Spectral_Contrast6', 'Spectral_Contrast7', 
                        'Spectral_Flatness']
    
    columns = ['Class'] + mfcc_columns + chroma_columns + spectral_columns
    return pd.DataFrame(features, columns=columns)

# Assuming your dataset is stored in 'dataset_path'
dataset_path = 'data'
audio_features_df = extract_audio_features_with_filters(dataset_path)

# Displaying the first few rows of the extracted features
print(audio_features_df.head())