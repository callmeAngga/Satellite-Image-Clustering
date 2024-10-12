import streamlit as st
import numpy as np
from PIL import Image
from src.kmeans import KMeans
from src.preprocessing import preprocess_image, extract_features, clean_small_regions
from src.visualization import visualize_results, visualize_cluster_info
import cv2

# Fungsi untuk memeriksa apakah warna adalah hijau
def is_green(rgb_color):
    # Konversi warna RGB ke format HSV untuk pemeriksaan warna
    hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
    
    # Definisikan rentang warna hijau dalam format HSV
    lower_green = np.array([30, 50, 50]) 
    upper_green = np.array([90, 255, 255])
    
    # Kembalikan True jika warna berada dalam rentang hijau
    return np.all(lower_green <= hsv_color) and np.all(hsv_color <= upper_green)

# Fungsi untuk memeriksa apakah warna RGB adalah hijau gelap
def is_dark_green(rgb_color):
    r, g, b = rgb_color
    return g > r and g > b and g > 50 and g < 100

# Fungsi untuk memeriksa apakah warna RGB adalah warna bangunan
def is_building(rgb_color):
    r, g, b = rgb_color

    # Periksa kondisi spesifik untuk warna bangunan
    return (
        (r > 150 and g > 120 and b > 100) or  # Warna bangunan terang
        (r == 208 and g == 143 and b == 118) or  # Warna spesifik genteng Indonesia (orange)
        (r == 179 and g == 175 and b == 166)  # Warna spesifik abu-abu bangunan
    )
    
# Fungsi untuk memeriksa apakah warna RGB adalah warna air
def is_water(rgb_color):
    r, g, b = rgb_color
    return (r < 120 and g < 120 and b < 120) or (r == 99 and g == 111 and b == 92)

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    # Atur pengaturan halaman aplikasi
    st.set_page_config(
        page_title="City Area Segmentation",        # Set judul aplikasi
        layout="wide"                               # Atur tata letak halamana menjadi lebar
    )    
    
    # Tampilkan judul dan penjelasan aplikasi
    st.title('Image Segmentation with K-Means Clustering')
    st.markdown("""
    This project determines whether an area is liveable based on two criteria:
    1. Green area > 30% of the total area, or
    2. Green area > Building area

    **Note:** This program requires input images from aerial/satellite/UAV imagery.
    """)
    st.markdown("---")

    # Buat dua kolom untuk antarmuka pengguna
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        n_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=4)

        # Jika file diunggah, tampilkan gambar
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

        process_button = st.button('Perform Clustering', type="primary")

    with col2:
        # Jika gambar diunggah dan tombol ditekan, lakukan pemrosesan
        if uploaded_file is not None and process_button:
            # Tampilkan animasi saat memproses
            with st.spinner("Processing image..."):
                original_image = np.array(image)                            # Konversi gambar ke array NumPy
                preprocessed_image = preprocess_image(original_image)       # Proses gambar
                features = extract_features(preprocessed_image)             # Ekstrak fitur dari gambar yang diproses

                # Inisialisasi KMeans dengan jumlah cluster yang dipilih
                kmeans = KMeans(n_clusters=n_clusters)
                
                # Lakukan clustering pada fitur
                labels = kmeans.fit_predict(features)

                segmented_image = labels.reshape(original_image.shape[:2])              # Ubah label menjadi citra tersegmentasi
                cleaned_image = clean_small_regions(segmented_image, min_size=200)      # Bersihkan area kecil dari hasil segmentasi

                st.subheader("Clustering Results")
                fig1, avg_colors = visualize_results(original_image, cleaned_image)     # Visualisasikan hasil clustering
                st.pyplot(fig1)                                                         

                fig2 = visualize_cluster_info(cleaned_image, avg_colors)                # Visualisasikan informasi cluster
                st.pyplot(fig2)
                
                # Identifikasi label hijau dari hasil clustering
                green_labels = [i for i, color in enumerate(avg_colors) if is_green(color) or is_dark_green(color)]
                
                # Hitung persentase area terbuka hijau
                if green_labels:
                    green_area_percentage = np.sum([np.sum(cleaned_image == label) for label in green_labels]) / cleaned_image.size * 100
                else:
                    green_area_percentage = 0.0
                
                # Identifikasi label non-hijau dan bangunan
                non_green_labels = [i for i in range(n_clusters) if i not in green_labels and not is_water(avg_colors[i])]
                building_labels = [i for i in non_green_labels if is_building(avg_colors[i])]
                building_area_percentage = np.sum([np.sum(cleaned_image == label) for label in building_labels]) / cleaned_image.size * 100 if building_labels else 0.0
                
                st.markdown("---")
                st.subheader("Area Analysis")
                col_green, col_building = st.columns(2)
                with col_green:
                    st.metric("Green Area", f"{green_area_percentage:.2f}%")
                    st.caption(f"From clusters: {[label + 1 for label in green_labels]}")
                with col_building:
                    st.metric("Building Area", f"{building_area_percentage:.2f}%")
                    st.caption(f"From clusters: {[label + 1 for label in building_labels]}")
                
                st.markdown("---")
                st.subheader("Livability Assessment")
                
                # Tentukan apakah area layak huni berdasarkan kriteria
                is_livable = green_area_percentage >= 30 or (green_area_percentage < 30 and green_area_percentage > building_area_percentage)
                        
                if is_livable:
                    st.success("This area is considered livable.")
                    if green_area_percentage >= 30 and green_area_percentage > building_area_percentage:
                        st.write("✅ Green area >= 30%")
                        st.write("✅ Green area > Building area")
                    elif green_area_percentage >= 30:
                        st.write("✅ Green area >= 30%")
                        st.write("ℹ️ Green area <= Building area")
                    else:
                        st.write("ℹ️ Green area < 30%")
                        st.write("✅ Green area > Building area")
                else:
                    st.error("This area is not considered livable.")
                    st.write("❌ Green area < 30%")
                    st.write("❌ Green area <= Building area")

# MFungsi Utama
if __name__ == "__main__":
    main()
