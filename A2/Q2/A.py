import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

# a

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


# b

import random

# Select 3 random classes
random_classes = random.sample(audio_features['Class'].unique().tolist(), 3)
print(f"Selected Classes: {random_classes}")

# Select 1 random file from each class
random_files = []
for cls in random_classes:
    class_folder = os.path.join(dataset_path, cls)
    # Filter to include only .wav files
    files_in_class = [f for f in os.listdir(class_folder) if f.endswith('.wav')]
    if files_in_class:  # Ensure the list is not empty
        selected_file = random.choice(files_in_class)
        file_path = os.path.join(class_folder, selected_file)
        random_files.append(file_path)

print(random_files)


def plot_audio_representation(file_path, title):
    print(f"Loading file: {file_path}")
    y, sr = librosa.load(file_path, sr=16000)

    plt.figure(figsize=(14, 8))

    # Plot Waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform of {title}')

    # Plot Spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.title(f'Spectrogram of {title}')
    plt.colorbar(format='%+2.0f dB')

    # Plot Mel-Spectrogram
    plt.subplot(3, 1, 3)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.title(f'Mel-Spectrogram of {title}')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

# Plot for each of the randomly selected files
for i, file in enumerate(random_files):
    plot_audio_representation(file, f'File {i+1} from {random_classes[i]}')


# c

# Count the number of samples per class
class_distribution = audio_features['Class'].value_counts()

# Plot the distribution
plt.figure(figsize=(12, 6))
class_distribution.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=90)
plt.show()