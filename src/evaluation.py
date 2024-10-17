import numpy as np
from sklearn.metrics import silhouette_score

def evaluate_clustering(features, labels):
    # Hitung silhouette score (mengukur seberapa baik klasterisasi)
    silhouette_avg = silhouette_score(features, labels)
    
    # Hitung inertia (jumlah kuadrat jarak titik ke pusat kluster)
    unique_labels = np.unique(labels)  # Ambil label kluster unik
    inertia = 0
    for label in unique_labels:
        cluster_points = features[labels == label]  # Ambil titik dalam kluster
        cluster_center = np.mean(cluster_points, axis=0)  # Hitung pusat kluster
        inertia += np.sum((cluster_points - cluster_center) ** 2)  # Tambah jarak kuadrat ke inertia
    
    return {
        'silhouette_score': silhouette_avg,  # Kembalikan silhouette score
        'inertia': inertia  # Kembalikan inertia
    }
