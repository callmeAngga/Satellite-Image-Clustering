import cv2
import numpy as np

def preprocess_image(image):
    # Ubah gambar menjadi float32 dan normalisasi ke rentang 0-1.
    image = image.astype(np.float32) / 255.0
    
    # Terapkan Gaussian blur untuk mengurangi noise di gambar.
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Ubah gambar ke LAB color space untuk analisis warna yang lebih baik.
    lab_image = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    
    return lab_image

def extract_features(image):
    # Ambil ukuran gambar, kita butuh ini untuk manipulasi.
    rows, cols = image.shape[:2]
    
    # Ubah gambar menjadi array 2D yang hanya berisi fitur warna (RGB).
    color_features = image.reshape(-1, 3)
    
    # Buat grid koordinat untuk setiap piksel.
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    
    # Gabungkan fitur warna dengan koordinat piksel untuk fitur spasial.
    spatial_features = np.column_stack((x_coords.ravel() / cols, y_coords.ravel() / rows))
    features = np.column_stack((color_features, spatial_features))
    return features

def clean_small_regions(labels, min_size):
    # Siapkan array kosong untuk menyimpan label yang sudah dibersihkan.
    cleaned_labels = np.zeros_like(labels)
    
    # Periksa setiap label unik yang kita punya.
    for label in range(np.max(labels) + 1):
        # Buat mask untuk menandai area dengan label saat ini.
        mask = (labels == label).astype(np.uint8)
        
        # Temukan komponen terhubung di area yang di-mask.
        num_components, components = cv2.connectedComponents(mask)
        
        # Lihat setiap komponen yang ditemukan.
        for i in range(1, num_components):
            component_mask = (components == i)
            
            # Jika ukuran komponen lebih besar dari min_size, simpan label itu.
            if np.sum(component_mask) > min_size:
                cleaned_labels[component_mask] = label
    return cleaned_labels

