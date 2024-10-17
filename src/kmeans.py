import numpy as np
import pickle

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters  # Tentukan jumlah kluster
        self.max_iters = max_iters  # Tentukan jumlah iterasi maksimal
        self.centroids = None  # Inisialisasi centroid

    def fit(self, X):
        # Inisialisasi centroid secara acak dari data
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iters):
            # Mengelompokkan titik berdasarkan centroid terdekat
            labels = self._assign_clusters(X)

            # Memperbarui centroid dengan menghitung rata-rata titik di setiap kluster
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Cek apakah centroid sudah konvergen
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids  # Perbarui centroid

    def predict(self, X):
        return self._assign_clusters(X)  # Mengelompokkan data baru

    def _assign_clusters(self, X):
        # Hitung jarak antara titik dan centroid
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        # Mengelompokkan setiap titik ke centroid terdekat
        return np.argmin(distances, axis=0)

    def save_model(self, filename):
        # Simpan model (centroid) ke file menggunakan pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.centroids, f)

    @classmethod
    def load_model(cls, filename):
        kmeans = cls()  # Buat objek KMeans baru
        # Muat model (centroid) dari file
        with open(filename, 'rb') as f:
            kmeans.centroids = pickle.load(f)
        return kmeans  # Kembalikan objek KMeans yang sudah dimuat

    @property
    def cluster_centers_(self):
        return self.centroids  # Mengembalikan centroid kluster
