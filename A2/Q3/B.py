import cv2
import pandas as pd
import os
from skimage.feature import hog
import matplotlib.pyplot as plt

os.makedirs('Plots', exist_ok=True)

labels_df = pd.read_csv('label.csv')

image_directory = 'data'

# Extract HOG features
def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys',
                              visualize=True)
    return features

# Extract color histograms
def extract_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_and_plot_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    
    hist = cv2.normalize(hist, hist).flatten()
    
    h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(h_hist, color='r')
    plt.title('Hue Channel Histogram')
    plt.xlim([0, 180])

    plt.subplot(1, 3, 2)
    plt.plot(s_hist, color='g')
    plt.title('Saturation Channel Histogram')
    plt.xlim([0, 256])

    plt.subplot(1, 3, 3)
    plt.plot(v_hist, color='b')
    plt.title('Value Channel Histogram')
    plt.xlim([0, 256])

    plt.tight_layout()

    plot_path = os.path.join('Plots', filename)
    plt.savefig(plot_path)
    plt.close()

with open('features_extraction.txt', 'w') as f:
    
    f.write("HOG Features and Color Histogram Features for Each Image:\n\n")

    for index, row in labels_df.iterrows():
        image_path = os.path.join(image_directory, row['filename'])
        
        # Extract HOG features
        hog_features = extract_hog_features(image_path)
        if hog_features is not None:
            f.write(f"Image: {row['filename']} (HOG Features)\n")
            f.write(f"{hog_features[:10]} ... [Truncated]\n")
        else:
            f.write(f"Image: {row['filename']} (HOG Features) - Failed to extract\n")
        
        # Extract color histogram
        color_histogram = extract_color_histogram(image_path)
        if color_histogram is not None:
            f.write(f"Image: {row['filename']} (Color Histogram Features)\n")
            f.write(f"{color_histogram[:10]} ... [Truncated]\n")
        else:
            f.write(f"Image: {row['filename']} (Color Histogram Features) - Failed to extract\n")
        
        extract_and_plot_color_histogram(image_path, f"{row['filename']}_histogram.png")

        f.write("\n")

print("Feature extraction and plotting complete. Plots saved in 'Plots' directory.")