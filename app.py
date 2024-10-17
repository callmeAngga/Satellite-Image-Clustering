import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
from src.kmeans import KMeans
from src.preprocessing import preprocess_image, extract_features, clean_small_regions
from src.visualization import visualize_results, visualize_cluster_info
from src.evaluation import evaluate_clustering

# Fungsi untuk memeriksa apakah sebuah warna termasuk hijau
def is_green(rgb_color):
    # Mengonversi warna RGB ke HSV untuk deteksi warna yang lebih akurat
    hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([90, 255, 255])
    return np.all(lower_green <= hsv_color) and np.all(hsv_color <= upper_green)

# Fungsi untuk memeriksa apakah sebuah warna termasuk hijau gelap
def is_dark_green(rgb_color):
    # Mengonversi warna RGB ke HSV dan memeriksa nilai-nilai spesifik untuk hijau gelap
    hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
    return hsv_color[0] > 60 and hsv_color[1] > 40 and hsv_color[2] < 100

# Fungsi untuk memeriksa apakah sebuah warna termasuk warna bangunan
def is_building(rgb_color):
    r, g, b = rgb_color
    # Memeriksa beberapa kondisi warna yang biasanya mewakili bangunan
    return (
        (r > 150 and g > 120 and b > 100) or
        (r == 208 and g == 143 and b == 118) or
        (r == 179 and g == 175 and b == 166)
    )

# Fungsi untuk mengubah label klaster menjadi gambar RGB
def labels_to_rgb(labels, centroids):
    height, width = labels.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Pastikan centroid memiliki 3 saluran (RGB)
    if centroids.shape[1] == 1:  # Jika centroid grayscale
        centroids = np.repeat(centroids, 3, axis=1)  # Ubah menjadi RGB dengan mengulang

    # Tetapkan warna centroid ke masing-masing label klaster
    for k in range(centroids.shape[0]):
        rgb_image[labels == k] = centroids[k, :3]  # Tetapkan warna centroid ke setiap label (RGB)

    return rgb_image

# Fungsi untuk melatih model KMeans dengan gambar yang diunggah
def train_model(n_clusters, uploaded_images):
    st.write("Training KMeans model...")

    if not uploaded_images:
        st.error("No images uploaded for training!")
        return None

    # Memuat dan memproses gambar latih dari input pengguna
    train_images = []
    for uploaded_image in uploaded_images:
        img = Image.open(uploaded_image)
        train_images.append(np.array(img))

    # Ekstraksi fitur dari semua gambar latih
    all_features = []
    for i, img in enumerate(train_images):
        st.write(f"Processing training image {i+1}/{len(train_images)}...")
        preprocessed = preprocess_image(img)
        features = extract_features(preprocessed)
        all_features.append(features)

        # Melatih model KMeans pada fitur gambar ini
        if len(features) > 0:
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(features)

            # Lakukan segmentasi
            segmented_image = kmeans.predict(features).reshape(img.shape[:2])
            cleaned_image = clean_small_regions(segmented_image, min_size=100)  # Membersihkan area kecil

            # Visualisasi hasil segmentasi untuk setiap gambar latih
            fig, avg_colors = visualize_results(img, cleaned_image)
            st.pyplot(fig)  # Tampilkan hasil visualisasi
            
    # Gabungkan semua fitur dari gambar latih
    combined_features = np.vstack(all_features)

    # Melatih model KMeans pada fitur gabungan
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(combined_features)

    # Simpan model yang dilatih
    kmeans.save_model('models/kmeans_model.pkl')
    st.write("Model training completed and saved.")

    return kmeans

# Fungsi utama aplikasi
def main():
    st.set_page_config(page_title="City Area Segmentation", layout="wide")

    st.title('Image Segmentation with K-Means Clustering')
    
    # Dropdown untuk memilih mode klasterisasi
    option = st.selectbox("Select Clustering Mode", ["Clustering Biasa", "Clustering Kelayakan Huni Suatu Area"])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Train Clustering Model")
        n_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=4)

        # Berikan info berbeda berdasarkan mode klasterisasi
        if option == "Clustering Kelayakan Huni Suatu Area":
            st.info("""
            Upload satellite images for training. Images should:
            - Be aerial/satellite views
            - Include various urban areas with different vegetation/building ratios
            - Be clear enough to distinguish vegetation and buildings
            """)
        else:
            st.info("""
            Upload images for training. Images should:
            - Be relevant to each other and come from consistent data sources
            - Represent various environments or categories based on the features you want to cluster
            """)

        # Unggah gambar latih untuk model
        uploaded_train_files = st.file_uploader("Upload training images...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

        # Tombol untuk memulai pelatihan model
        if st.button('Train Model'):
            train_model(n_clusters, uploaded_train_files)

        st.subheader("Image Processing")
        uploaded_file = st.file_uploader("Choose an image to process...", type=["jpg", "jpeg", "png"])

        # Jika gambar diunggah, tampilkan gambar dan tombol untuk memproses
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            process_button = st.button('Process Image', type="primary")

    with col2:
        if uploaded_file is not None and process_button:
            with st.spinner("Processing image..."):
                # Muat model yang telah dilatih
                kmeans = KMeans.load_model('models/kmeans_model.pkl')

                # Proses gambar dengan model
                original_image = np.array(image)
                preprocessed_image = preprocess_image(original_image)
                features = extract_features(preprocessed_image)

                # Lakukan klasterisasi
                labels = kmeans.predict(features)
                segmented_image = labels.reshape(original_image.shape[:2])
                cleaned_image = clean_small_regions(segmented_image, min_size=200)

                # Visualisasi hasil klasterisasi
                st.subheader("Clustering Results")
                fig1, avg_colors = visualize_results(original_image, cleaned_image)
                st.pyplot(fig1)
                fig2 = visualize_cluster_info(cleaned_image, avg_colors)
                st.pyplot(fig2)
                
                # Analisis kelayakan huni jika mode yang dipilih
                if option == "Clustering Kelayakan Huni Suatu Area":
                    st.subheader("Area Analysis for Livability")
                    # Identifikasi label klaster yang mewakili area hijau
                    green_labels = [i for i, color in enumerate(avg_colors) if is_green(color) or is_dark_green(color)]
                    if green_labels:
                        green_area_percentage = np.sum([np.sum(cleaned_image == label) for label in green_labels]) / cleaned_image.size * 100
                    else:
                        green_area_percentage = 0.0

                    # Identifikasi label klaster yang mewakili bangunan
                    non_green_labels = [i for i in range(n_clusters) if i not in green_labels]
                    building_labels = [i for i in non_green_labels if is_building(avg_colors[i])]
                    if building_labels:
                        building_area_percentage = np.sum([np.sum(cleaned_image == label) for label in building_labels]) / cleaned_image.size * 100
                    else:
                        building_area_percentage = 0.0

                    # Tampilkan persentase area hijau dan bangunan
                    col_green, col_building = st.columns(2)
                    with col_green:
                        st.metric("Green Area", f"{green_area_percentage:.2f}%")
                    with col_building:
                        st.metric("Building Area", f"{building_area_percentage:.2f}%")

                    # Penilaian kelayakan huni
                    st.subheader("Livability Assessment")
                    is_livable = green_area_percentage >= 30 or (green_area_percentage < 30 and green_area_percentage > building_area_percentage)

                    if is_livable:
                        st.success("This area is considered livable.")
                    else:
                        st.error("This area is not considered livable.")

                else:
                    # Ringkasan klasterisasi untuk opsi biasa
                    st.subheader("Clustering Summary")
                    st.write(f"Number of Clusters: {n_clusters}")
                    st.write("Cluster analysis completed. For a more detailed analysis, choose the livability option.")

if __name__ == "__main__":
    main()