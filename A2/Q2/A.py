import os
import librosa
import numpy as np
import pandas as pd

# Function to extract audio features
def extract_audio_features(directory):
    features = []
    for foldername in os.listdir(directory):
        folder_path = os.path.join(directory, foldername)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(folder_path, filename)
                    # Load the audio file
                    y, sr = librosa.load(file_path, sr=16000)
                    # Calculate duration
                    duration = librosa.get_duration(y=y, sr=sr)
                    # Store amplitude values (we can use the mean amplitude for simplicity)
                    amplitude = np.mean(np.abs(y))
                    # Append to list
                    features.append([foldername, amplitude, duration])
    return pd.DataFrame(features, columns=['Class', 'Amplitude', 'Duration'])

# Assuming your dataset is stored in 'dataset_path'
dataset_path = 'data'
audio_features = extract_audio_features(dataset_path)

# Displaying the first few rows of the extracted features
print(audio_features.head())

# Calculate statistical summary
summary = audio_features.groupby('Class').agg(
    Amplitude_Mean=('Amplitude', 'mean'),
    Amplitude_Std=('Amplitude', 'std'),
    Amplitude_Min=('Amplitude', 'min'),
    Amplitude_Max=('Amplitude', 'max'),
    Duration_Mean=('Duration', 'mean'),
    Duration_Std=('Duration', 'std'),
    Duration_Min=('Duration', 'min'),
    Duration_Max=('Duration', 'max')
)

# Display the summary
print(summary)