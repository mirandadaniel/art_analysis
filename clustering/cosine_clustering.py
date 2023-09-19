import os
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    filenames = []
    images = []

    image_filenames = os.listdir(folder)

    for filename in image_filenames:
        with Image.open(os.path.join(folder, filename)) as img:
            if img is not None:
                images.append(img.copy())
                filenames.append(filename)

    return images, filenames

def preprocess_images(images):
    resized_images = [img.resize((100, 100)) for img in images]
    flattened_images = [np.array(img).flatten() for img in resized_images]
    return np.array(flattened_images)

def main():
    folder_path = "/art_class/style_hm"
    images, filenames = load_images_from_folder(folder_path)
    print("filenames are: ", filenames)

    X = preprocess_images(images)

    num_clusters = 6  # Set the desired number of clusters

    cosine_distances_matrix = cosine_distances(X) 

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_assignments = kmeans.fit_predict(cosine_distances_matrix)

    cluster_heatmaps = {cluster: [] for cluster in range(num_clusters)}
    cluster_filenames = {cluster: [] for cluster in range(num_clusters)}

    for i, cluster in enumerate(cluster_assignments):
        cluster_heatmaps[cluster].append(images[i])
        cluster_filenames[cluster].append(filenames[i])

    output_text_file = "/art_class/clustering/cosine_clusters/cluster_info.txt" 

    with open(output_text_file, "w") as f:
        for cluster, heatmaps in cluster_heatmaps.items():
            f.write(f"Cluster {cluster + 1} contains {len(heatmaps)} heatmaps:\n")
            for i, heatmap in enumerate(heatmaps):
                filename = cluster_filenames[cluster][i]
                f.write(f"Heatmap {i + 1}: {filename}\n")

    output_folder = "/art_class/clustering/cosine_clusters/top_heatmaps/"

    for cluster, heatmaps in cluster_heatmaps.items():
        os.makedirs(output_folder, exist_ok=True)
        for i, heatmap in enumerate(heatmaps[:5]):
            original_filename = cluster_filenames[cluster][i]
            original_filename_without_extension = os.path.splitext(original_filename)[0]
            new_filename = f"cluster_{cluster + 1}_heatmap_{i + 1}_{original_filename_without_extension}.jpg"
            heatmap.save(os.path.join(output_folder, new_filename))

    print("Top heatmap images saved.")

if __name__ == "__main__":
    main()
