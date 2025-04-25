import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import datetime
warnings.filterwarnings('ignore')

def analyze_prelabels(data_file, output_dir=None):
    """
    Farklı prelabel kombinasyonlarını test eden ve silhouette skorunu iyileştirmeye çalışan fonksiyon
    
    Parameters:
    -----------
    data_file : str
        Müşteri verisini içeren CSV dosyasının yolu
    output_dir : str, optional
        Sonuçların kaydedileceği dizin
    """
    # Çıktı dizini oluştur
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"prelabel_analysis_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Veriyi yükle
    data = pd.read_csv(data_file)
    print(f"Veri yüklendi: {data.shape[0]} müşteri, {data.shape[1]} özellik")
    
    # Sayısal kümeleme özellikleri
    numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    
    # Verileri ölçekle
    scaler = StandardScaler()
    X = data[numerical_features].values
    X_scaled = scaler.fit_transform(X)
    
    # Kullanıcının belirttiği prelabel kombinasyonları
    prelabel_combinations = [
        ['Location', 'Payment Method'],
        ['Location', 'Age', 'Gender'],
        ['Payment Method', 'Shipping Type', 'Frequency of Purchases'],
        ['Location', 'Season'],
        ['Color', 'Season'],
        ['Location', 'Frequency of Purchases'],
        ['Location', 'Frequency of Purchases', 'Category'],
        ['Location', 'Shipping Type'],
        ['Gender', 'Shipping Type'],
        ['Gender', 'Item Purchased'],
        ['Review Rating', 'Item Purchased']
    ]
    
    # Sonuçları depolamak için liste
    results = []
    improvements = []
    
    # 1. Önce baseline skorunu hesapla (prelabel olmadan)
    n_clusters = 4  # Varsayılan küme sayısı
    kmeans_baseline = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    baseline_labels = kmeans_baseline.fit_predict(X_scaled)
    baseline_score = silhouette_score(X_scaled, baseline_labels)
    print(f"Baseline Silhouette skoru (prelabel olmadan): {baseline_score:.4f}")
    
    # Her bir kombinasyon için
    for combo in prelabel_combinations:
        combo_name = " + ".join(combo)
        print(f"\nPrelabel kombinasyonu test ediliyor: {combo_name}")
        
        # Bu kombinasyonda eksik sütun var mı kontrol et
        missing_columns = [col for col in combo if col not in data.columns]
        if missing_columns:
            print(f"  Uyarı: {missing_columns} sütunları veride bulunmuyor. Bu kombinasyon atlanıyor.")
            continue
        
        # Kombinasyondaki her sütun için etiketleri birleştir
        # Örnek: "New York-Credit Card-Male" gibi bir etiket oluşturulur
        prelabel_values = data[combo[0]].astype(str)
        for col in combo[1:]:
            prelabel_values = prelabel_values + "-" + data[col].astype(str)
        
        # Eşsiz değerleri bul
        unique_prelabels = prelabel_values.unique()
        n_unique_prelabels = len(unique_prelabels)
        print(f"  Bu kombinasyonda {n_unique_prelabels} eşsiz prelabel var")
        
        # Eğer çok az prelabel varsa, bu kombinasyonu atla
        if n_unique_prelabels < 2:
            print("  Çok az eşsiz prelabel. Bu kombinasyon atlanıyor.")
            continue
        
        # Kaç küme kullanılacak?
        # Eğer çok fazla eşsiz değer varsa, küme sayısını sınırla
        if n_unique_prelabels > 20:
            effective_n_clusters = min(n_unique_prelabels, 10)
        else:
            effective_n_clusters = n_unique_prelabels
        
        print(f"  {effective_n_clusters} küme kullanılıyor")
        
        # Prelabel değerlerini sayısal değerlere dönüştür
        label_encoder = LabelEncoder()
        encoded_prelabels = label_encoder.fit_transform(prelabel_values)
        
        # Her eşsiz prelabel için centroids oluştur
        centroids = np.zeros((effective_n_clusters, X_scaled.shape[1]))
        
        # Eğer prelabel sayısı istenen küme sayısından azsa
        if n_unique_prelabels < effective_n_clusters:
            for i, label in enumerate(label_encoder.classes_):
                mask = (prelabel_values == label)
                if np.sum(mask) > 0:
                    centroids[i] = X_scaled[mask].mean(axis=0)
            
            # Kalan küme merkezleri için k-means++ başlangıç kullan
            kmeans = KMeans(n_clusters=effective_n_clusters, random_state=42, n_init=10)
            
        # Eğer prelabel sayısı istenen küme sayısından fazlaysa, benzer olanları birleştir
        elif n_unique_prelabels > effective_n_clusters:
            # Her prelabel için merkez hesapla
            prelabel_centroids = np.zeros((n_unique_prelabels, X_scaled.shape[1]))
            for i, label in enumerate(label_encoder.classes_):
                mask = (prelabel_values == label)
                if np.sum(mask) > 0:
                    prelabel_centroids[i] = X_scaled[mask].mean(axis=0)
            
            # Prelabel merkezlerini kümeleme
            meta_kmeans = KMeans(n_clusters=effective_n_clusters, random_state=42, n_init=10)
            meta_labels = meta_kmeans.fit_predict(prelabel_centroids)
            
            # Merge edilmiş küme merkezlerini al
            for i in range(effective_n_clusters):
                mask = meta_labels == i
                if np.sum(mask) > 0:
                    selected_centroids = prelabel_centroids[mask]
                    # Ağırlıklı ortalama (centroid başına düşen örnek sayısına göre)
                    weights = np.array([np.sum(prelabel_values == label) for label in label_encoder.classes_[mask]])
                    centroids[i] = np.average(selected_centroids, axis=0, weights=weights)
            
            # Bu birleştirilmiş merkezleri kullan
            kmeans = KMeans(n_clusters=effective_n_clusters, init=centroids, n_init=1, random_state=42)
            
        # Prelabel sayısı istenen küme sayısıyla aynıysa
        else:
            for i, label in enumerate(label_encoder.classes_):
                mask = (prelabel_values == label)
                if np.sum(mask) > 0:
                    centroids[i] = X_scaled[mask].mean(axis=0)
            
            # Prelabel'lardan elde edilen merkezleri kullan
            kmeans = KMeans(n_clusters=effective_n_clusters, init=centroids, n_init=1, random_state=42)
        
        # KMeans uygula
        guided_labels = kmeans.fit_predict(X_scaled)
        
        # Silhouette skorunu hesapla
        guided_score = silhouette_score(X_scaled, guided_labels)
        improvement = guided_score - baseline_score
        
        print(f"  Silhouette skoru: {guided_score:.4f} (Baseline'dan fark: {improvement:.4f})")
        
        # Sonuçları kaydet
        results.append({
            'Prelabel_Combination': combo_name,
            'Num_Unique_Prelabels': n_unique_prelabels,
            'Num_Clusters': effective_n_clusters,
            'Silhouette_Score': guided_score,
            'Improvement': improvement
        })
        
        # Eğer iyileştirme varsa, etiketleri ve bunlara dayalı analizleri kaydet
        if improvement > 0:
            improvements.append({
                'combo': combo,
                'combo_name': combo_name,
                'score': guided_score,
                'improvement': improvement,
                'labels': guided_labels,
                'n_clusters': effective_n_clusters
            })
            
            # Kümelenmiş veriyi kaydet
            result_data = data.copy()
            result_data[f'Cluster_{combo_name}'] = guided_labels
            result_data.to_csv(os.path.join(output_dir, f"clustered_by_{combo_name.replace(' + ', '_')}.csv"), index=False)
            
            # Kümelere göre temel istatistikleri hesapla ve görselleştir
            cluster_stats = result_data.groupby(f'Cluster_{combo_name}').agg({
                'Age': ['mean', 'count'],
                'Purchase Amount (USD)': ['mean', 'min', 'max'],
                'Review Rating': 'mean',
                'Previous Purchases': 'mean'
            }).round(2)
            
            # İstatistikleri dosyaya kaydet
            with open(os.path.join(output_dir, f"stats_{combo_name.replace(' + ', '_')}.txt"), 'w') as f:
                f.write(f"Prelabel Kombinasyonu: {combo_name}\n")
                f.write(f"Silhouette Skoru: {guided_score:.4f} (Baseline'dan iyileştirme: {improvement:.4f})\n\n")
                f.write("Küme İstatistikleri:\n")
                f.write(str(cluster_stats))
                
                f.write("\n\nKüme İçerikleri:\n")
                for cluster in range(effective_n_clusters):
                    f.write(f"\nKüme {cluster}:\n")
                    # Her küme için en sık bulunan değerleri göster
                    for col in combo:
                        if col in data.columns:
                            top_values = result_data[result_data[f'Cluster_{combo_name}'] == cluster][col].value_counts().head(5)
                            f.write(f"{col} için en sık değerler:\n{top_values}\n\n")
            
            # Kümeleri görselleştir (PCA veya t-SNE kullanılabilir)
            plt.figure(figsize=(10, 6))
            for cluster in range(effective_n_clusters):
                mask = guided_labels == cluster
                plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], label=f'Cluster {cluster}', alpha=0.7)
            
            plt.title(f'Clusters Using {combo_name} as Prelabels')
            plt.xlabel('Feature 1 (Scaled)')
            plt.ylabel('Feature 2 (Scaled)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"clusters_{combo_name.replace(' + ', '_')}.png"), dpi=300)
            plt.close()
    
    # Tüm sonuçları DataFrame'e dönüştür ve kaydet
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "all_prelabel_results.csv"), index=False)
    
    # Sonuçları görselleştir
    if len(results) > 0:
        plt.figure(figsize=(12, 8))
        results_sorted = results_df.sort_values('Silhouette_Score', ascending=False)
        
        # Baseline çizgisi
        plt.axhline(y=baseline_score, color='r', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_score:.4f})')
        
        # Silhouette skorlarını göster
        ax = sns.barplot(x='Prelabel_Combination', y='Silhouette_Score', data=results_sorted)
        plt.title('Silhouette Scores for Different Prelabel Combinations')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prelabel_comparison.png"), dpi=300)
        plt.close()
        
        # İyileştirmeleri göster
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Prelabel_Combination', y='Improvement', data=results_sorted)
        plt.title('Silhouette Score Improvements from Baseline')
        plt.xticks(rotation=90)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prelabel_improvements.png"), dpi=300)
        plt.close()
    
    # En iyi sonuçları göster
    print("\n=== Prelabel Analizi Sonuçları ===")
    if len(results) > 0:
        best_result = max(results, key=lambda x: x['Silhouette_Score'])
        print(f"En yüksek Silhouette skoru: {best_result['Silhouette_Score']:.4f}")
        print(f"En iyi prelabel kombinasyonu: {best_result['Prelabel_Combination']}")
        print(f"Baseline'dan iyileştirme: {best_result['Improvement']:.4f}")
        
        # En iyi 3 kombinasyonu yazdır
        print("\nEn iyi 3 Prelabel Kombinasyonu:")
        for i, result in enumerate(sorted(results, key=lambda x: x['Silhouette_Score'], reverse=True)[:3]):
            print(f"{i+1}. {result['Prelabel_Combination']}: {result['Silhouette_Score']:.4f} (Fark: {result['Improvement']:.4f})")
        
        # Baseline'dan daha kötü performans gösteren kombinasyonları yazdır
        worse_than_baseline = [r for r in results if r['Improvement'] < 0]
        if worse_than_baseline:
            print("\nBaseline'dan daha kötü performans gösteren kombinasyonlar:")
            for result in worse_than_baseline:
                print(f"- {result['Prelabel_Combination']}: {result['Silhouette_Score']:.4f} (Fark: {result['Improvement']:.4f})")
    else:
        print("Hiçbir prelabel kombinasyonu değerlendirilemedi.")
    
    print(f"\nTüm sonuçlar {output_dir} dizinine kaydedildi.")
    
    return results_df, improvements

if __name__ == "__main__":
    # Veriyi analiz et
    results, improvements = analyze_prelabels('customer_data.csv') 