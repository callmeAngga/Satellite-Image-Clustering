import matplotlib.pyplot as plt
import numpy as np

def visualize_results(original_image, segmented_image):
    # Jika gambar asli memiliki 4 channel (RGBA), gunakan hanya 3 channel (RGB)
    if original_image.shape[-1] == 4:
        original_image = original_image[..., :3]
    
    # Hitung jumlah kluster
    n_clusters = np.max(segmented_image) + 1
    avg_colors = np.zeros((n_clusters, 3), dtype=np.float32)  # Inisialisasi warna rata-rata untuk setiap kluster
    
    # Hitung warna rata-rata untuk setiap kluster
    for cluster in range(n_clusters):
        mask = (segmented_image == cluster)  # Mask untuk kluster tertentu
        if np.any(mask):
            avg_colors[cluster] = np.mean(original_image[mask], axis=0)  # Warna rata-rata kluster

    # Buat gambar tersegmentasi dengan warna rata-rata per kluster
    segmented_colored = np.zeros_like(original_image, dtype=np.float32)
    for cluster in range(n_clusters):
        mask = (segmented_image == cluster)
        segmented_colored[mask] = avg_colors[cluster]
    
    # Visualisasi gambar asli dan gambar tersegmentasi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    ax1.imshow(original_image.astype(np.uint8))  # Tampilkan gambar asli
    ax1.set_title('Original Image')  # Judul
    ax1.axis('off')  # Matikan axis
    
    ax2.imshow(segmented_colored.astype(np.uint8))  # Tampilkan gambar tersegmentasi
    ax2.set_title('Segmented Image')
    ax2.axis('off')

    plt.tight_layout()
    return fig, avg_colors  # Kembalikan figure dan warna rata-rata kluster

def visualize_cluster_info(segmented_image, avg_colors):
    # Hitung jumlah kluster
    n_clusters = np.max(segmented_image) + 1
    # Hitung ukuran setiap kluster
    cluster_sizes = [np.sum(segmented_image == i) for i in range(n_clusters)]
    total_pixels = np.prod(segmented_image.shape)  # Hitung total piksel
    # Hitung persentase setiap kluster
    cluster_percentages = [size / total_pixels * 100 for size in cluster_sizes]
    
    # Visualisasi informasi kluster
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(n_clusters), cluster_percentages, color=np.array(avg_colors) / 255.0)
    
    ax.set_xlabel('Cluster')  # Label sumbu x
    ax.set_ylabel('Percentage')  # Label sumbu y
    ax.set_title('Cluster Information')  # Judul
    ax.set_xticks(range(n_clusters))  # Setel tick pada sumbu x
    ax.set_xticklabels([f'Cluster {i+1}' for i in range(n_clusters)])  # Label kluster
    
    # Tambahkan teks persentase dan warna kluster di atas bar
    for i, (bar, percentage) in enumerate(zip(bars, cluster_percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%\n{tuple(np.round(avg_colors[i]).astype(int))}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig  # Kembalikan figure
