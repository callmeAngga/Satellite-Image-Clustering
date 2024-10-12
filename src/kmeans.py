import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        # Inisialisasi jumlah kluster dan maksimum iterasi
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit_predict(self, X):
        # Pilih centroid awal secara acak dari data
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        # Iterasi untuk memperbarui centroid dan label kluster
        for _ in range(self.max_iters):
            # Hitung jarak dari setiap titik ke setiap centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            
            # Tentukan label kluster berdasarkan centroid terdekat
            labels = np.argmin(distances, axis=0)
            
            # Hitung centroid baru dengan rata-rata titik dalam setiap kluster
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Jika centroid tidak berubah, keluar dari loop
            if np.all(self.centroids == new_centroids):
                break
            
            # Perbarui centroid untuk iterasi berikutnya
            self.centroids = new_centroids
        return labels