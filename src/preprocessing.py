import cv2
import numpy as np

def preprocess_image(image):
    # Konversi gambar ke tipe float32 dan normalisasi ke rentang 0-1
    image = image.astype(np.float32) / 255.0
    
    # Terapkan Gaussian blur untuk mengurangi noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Konversi gambar ke ruang warna LAB untuk analisis warna yang lebih baik
    lab_image = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    
    return lab_image

def extract_features(image):
    rows, cols = image.shape[:2]  # Dapatkan ukuran gambar
    
    # Ubah gambar menjadi array 2D dari fitur warna (RGB)
    color_features = image.reshape(-1, 3)
    
    # Buat grid koordinat untuk setiap piksel
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    
    # Gabungkan fitur warna dengan koordinat piksel untuk mendapatkan fitur spasial
    spatial_features = np.column_stack((x_coords.ravel() / cols, y_coords.ravel() / rows))
    features = np.column_stack((color_features, spatial_features))  # Gabungkan fitur warna dan spasial
    
    return features

def clean_small_regions(labels, min_size):
    cleaned_labels = np.zeros_like(labels)  # Inisialisasi label yang dibersihkan
    
    for label in range(np.max(labels) + 1):
        mask = (labels == label).astype(np.uint8)  # Buat mask untuk setiap label
        num_components, components = cv2.connectedComponents(mask)  # Temukan komponen yang terhubung
        
        for i in range(1, num_components):
            component_mask = (components == i)  # Buat mask untuk setiap komponen
            if np.sum(component_mask) > min_size:  # Hanya simpan komponen yang lebih besar dari ukuran minimum
                cleaned_labels[component_mask] = label  # Simpan label yang dibersihkan
    
    return cleaned_labels
