import matplotlib.pyplot as plt
import numpy as np

def visualize_results(original_image, segmented_image):
    # Hapus channel alpha jika ada
    if original_image.shape[-1] == 4:
        original_image = original_image[..., :3]

    # Hitung jumlah cluster
    n_clusters = np.max(segmented_image) + 1
    
    # Inisialisasi array untuk menyimpan warna rata-rata setiap cluster
    avg_colors = np.zeros((n_clusters, 3), dtype=np.float32)
    
    # Hitung warna rata-rata setiap cluster
    for cluster in range(n_clusters):
        mask = (segmented_image == cluster)
        if np.any(mask):
            avg_colors[cluster] = np.mean(original_image[mask], axis=0)

    # Buat gambar baru dengan warna rata-rata setiap cluster
    segmented_colored = np.zeros_like(original_image, dtype=np.float32)
    for cluster in range(n_clusters):
        mask = (segmented_image == cluster)
        segmented_colored[mask] = avg_colors[cluster]

    # Buat figure dan subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Tampilkan gambar asli
    ax1.imshow(original_image.astype(np.uint8))
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Tampilkan gambar yang telah disegmentasi
    ax2.imshow(segmented_colored.astype(np.uint8))
    ax2.set_title('Segmented Image')
    ax2.axis('off')

    plt.tight_layout()
    return fig, avg_colors

def visualize_cluster_info(segmented_image, avg_colors):
    # Hitung jumlah cluster
    n_clusters = np.max(segmented_image) + 1
    
    # Hitung jumlah pixel setiap cluster
    cluster_sizes = [np.sum(segmented_image == i) for i in range(n_clusters)]
    
    # Hitung persentase pixel setiap cluster
    total_pixels = np.prod(segmented_image.shape)
    cluster_percentages = [size / total_pixels * 100 for size in cluster_sizes]
    
    # Buat figure dan subplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Buat bar chart
    bars = ax.bar(range(n_clusters), cluster_percentages, color=np.array(avg_colors) / 255.0)
    
    # Set label dan judul
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Percentage')
    ax.set_title('Cluster Information')
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f'Cluster {i+1}' for i in range(n_clusters)])
    
    # Tambahkan label persentase dan warna rata-rata pada setiap bar
    for i, (bar, percentage) in enumerate(zip(bars, cluster_percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%\n{tuple(np.round(avg_colors[i]).astype(int))}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

