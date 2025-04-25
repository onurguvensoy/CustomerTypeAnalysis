import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import os
import datetime
warnings.filterwarnings('ignore')

def improve_silhouette(data_file, output_dir=None):
    """
    Daha spesifik yaklaşımlar kullanarak silhouette skorunu 0.5 üzerine çıkarmaya çalışan fonksiyon
    
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
        output_dir = f"improved_clustering_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Veriyi yükle
    data = pd.read_csv(data_file)
    print(f"Veri yüklendi: {data.shape[0]} müşteri, {data.shape[1]} özellik")
    
    # Feature selection aşaması: Farklı özellik kombinasyonları deneyin
    # Yani sadece belli sayısal özelliklere odaklanın, hepsini kullanmayın
    
    # Tüm sayısal özellikleri belirle
    all_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    
    # Farklı normalizasyon yöntemlerini test et
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
    }
    
    # Farklı özellik kombinasyonlarını test et
    feature_combinations = [
        ['Age', 'Purchase Amount (USD)'],  # Sadece yaş ve satın alma tutarı
        ['Purchase Amount (USD)', 'Previous Purchases'],  # Alım tutarı ve önceki alımlar
        ['Age', 'Review Rating'],  # Yaş ve değerlendirme puanı
        ['Review Rating', 'Previous Purchases'],  # Değerlendirme ve önceki alımlar
        ['Age', 'Purchase Amount (USD)', 'Previous Purchases'],  # 3lü kombinasyon
        all_features  # Tüm özellikler
    ]
    
    # Farklı kümeleme algoritmalarını test et
    clustering_methods = {
        'KMeans': lambda n: KMeans(n_clusters=n, random_state=42, n_init=10),
        'AgglomerativeClustering': lambda n: AgglomerativeClustering(n_clusters=n)
    }
    
    # DBSCAN için farklı parametreleri test et
    dbscan_params = [
        {'eps': 0.3, 'min_samples': 5},
        {'eps': 0.5, 'min_samples': 5},
        {'eps': 0.7, 'min_samples': 5},
        {'eps': 0.3, 'min_samples': 10},
        {'eps': 0.5, 'min_samples': 10},
        {'eps': 0.7, 'min_samples': 10}
    ]
    
    # Sonuçları depolamak için liste
    results = []
    best_config = None
    best_score = 0
    best_labels = None
    
    # 1. Farklı küme sayıları, özellik kombinasyonları ve normalizasyon yöntemleri dene
    for feature_set in feature_combinations:
        feature_str = '+'.join(feature_set)
        print(f"\nÖzellik seti test ediliyor: {feature_str}")
        
        # Seçilen özellikleri ayır
        X = data[feature_set].values
        
        # Her bir normalizasyon yöntemi için
        for scaler_name, scaler in scalers.items():
            # Verileri ölçekle
            X_scaled = scaler.fit_transform(X)
            
            # PCA uygula - hem 2 bileşenli hem de özellik sayısına eşit bileşen sayısıyla dene
            pca_components = [min(2, len(feature_set)), min(len(feature_set), len(feature_set))]
            
            for n_components in pca_components:
                # Eğer PCA uygulanacaksa
                if n_components < len(feature_set):
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(X_scaled)
                    analysis_data = X_pca
                    method_suffix = f"_PCA{n_components}"
                else:
                    analysis_data = X_scaled
                    method_suffix = ""
                
                # Farklı küme sayılarını dene
                for n_clusters in range(2, 11):
                    # KMeans ve Agglomerative için
                    for method_name, clustering_func in clustering_methods.items():
                        try:
                            # Kümeleme algoritmasını uygula
                            clusterer = clustering_func(n_clusters)
                            cluster_labels = clusterer.fit_predict(analysis_data)
                            
                            # En az 2 küme ve her kümede en az 1 örnek olmalı
                            if len(np.unique(cluster_labels)) < 2:
                                continue
                                
                            # Silhouette skorunu hesapla
                            silhouette_avg = silhouette_score(analysis_data, cluster_labels)
                            
                            # Sonucu kaydet
                            config_name = f"{method_name}_{scaler_name}_{feature_str}{method_suffix}_k{n_clusters}"
                            
                            results.append({
                                'Config': config_name,
                                'Features': feature_str,
                                'Scaler': scaler_name,
                                'Method': method_name,
                                'PCA_Components': n_components if n_components < len(feature_set) else None,
                                'Clusters': n_clusters,
                                'Silhouette': silhouette_avg
                            })
                            
                            print(f"  {config_name}: {silhouette_avg:.4f}")
                            
                            # En iyi sonucu güncelle
                            if silhouette_avg > best_score:
                                best_score = silhouette_avg
                                best_config = {
                                    'config_name': config_name,
                                    'features': feature_set,
                                    'scaler': scaler,
                                    'method': method_name,
                                    'pca_components': n_components if n_components < len(feature_set) else None,
                                    'clusters': n_clusters
                                }
                                best_labels = cluster_labels
                        except Exception as e:
                            print(f"  Hata: {method_name} (k={n_clusters}): {str(e)}")
                
                # DBSCAN için farklı parametreleri dene
                for params in dbscan_params:
                    try:
                        dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
                        db_labels = dbscan.fit_predict(analysis_data)
                        
                        # Gürültü noktaları hariç en az 2 küme ve her kümede en az 1 örnek olmalı
                        unique_labels = np.unique(db_labels)
                        if len(unique_labels) < 2 or (len(unique_labels) == 2 and -1 in unique_labels):
                            continue
                        
                        # Gürültü noktalarını (-1 etiketli) hariç tut
                        valid_indices = db_labels != -1
                        if np.sum(valid_indices) < 2:
                            continue
                            
                        # Silhouette skorunu hesapla (gürültü noktaları hariç)
                        silhouette_avg = silhouette_score(analysis_data[valid_indices], db_labels[valid_indices])
                        
                        # Sonucu kaydet
                        config_name = f"DBSCAN_{scaler_name}_{feature_str}{method_suffix}_eps{params['eps']}_ms{params['min_samples']}"
                        
                        results.append({
                            'Config': config_name,
                            'Features': feature_str,
                            'Scaler': scaler_name,
                            'Method': 'DBSCAN',
                            'PCA_Components': n_components if n_components < len(feature_set) else None,
                            'Clusters': len(np.unique(db_labels)) - (1 if -1 in db_labels else 0),
                            'Silhouette': silhouette_avg,
                            'DBSCAN_eps': params['eps'],
                            'DBSCAN_min_samples': params['min_samples'],
                            'Noise_points': np.sum(db_labels == -1)
                        })
                        
                        print(f"  {config_name}: {silhouette_avg:.4f} (Gürültü: {np.sum(db_labels == -1)})")
                        
                        # En iyi sonucu güncelle
                        if silhouette_avg > best_score:
                            best_score = silhouette_avg
                            best_config = {
                                'config_name': config_name,
                                'features': feature_set,
                                'scaler': scaler,
                                'method': 'DBSCAN',
                                'pca_components': n_components if n_components < len(feature_set) else None,
                                'dbscan_eps': params['eps'],
                                'dbscan_min_samples': params['min_samples']
                            }
                            best_labels = db_labels
                    except Exception as e:
                        pass
    
    # 2. Spesifik kategorik sütunlar için alt kümeleri ayırarak kümeleme dene
    # Örneğin: Sadece belirli bir yaş grubundaki veya belirli bir bölgedeki müşteriler
    age_groups = [(18, 30), (31, 45), (46, 60), (61, 100)]
    
    for age_min, age_max in age_groups:
        # Yaş grubunu filtrele
        age_filter = (data['Age'] >= age_min) & (data['Age'] <= age_max)
        if np.sum(age_filter) < 10:  # Çok az veri varsa atla
            continue
            
        subset_data = data[age_filter]
        print(f"\nYaş grubu {age_min}-{age_max} için kümeleme deneniyor ({len(subset_data)} müşteri)")
        
        # Bu alt küme için en iyi özellik kombinasyonlarını dene
        for feature_set in feature_combinations[:3]:  # İlk 3 kombinasyonu dene
            feature_str = '+'.join(feature_set)
            
            # Seçilen özellikleri ayır
            X = subset_data[feature_set].values
            
            # StandardScaler kullan
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # KMeans için farklı küme sayılarını dene
            for n_clusters in range(2, min(8, len(subset_data) // 5)):  # Alt küme boyutuna göre sınırla
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    
                    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                    
                    # Sonucu kaydet
                    config_name = f"KMeans_Age{age_min}-{age_max}_{feature_str}_k{n_clusters}"
                    
                    results.append({
                        'Config': config_name,
                        'Features': feature_str,
                        'Scaler': 'StandardScaler',
                        'Method': 'KMeans',
                        'Age_Group': f"{age_min}-{age_max}",
                        'Clusters': n_clusters,
                        'Silhouette': silhouette_avg,
                        'Subset_Size': len(subset_data)
                    })
                    
                    print(f"  {config_name}: {silhouette_avg:.4f}")
                    
                    # En iyi sonucu güncelle
                    if silhouette_avg > best_score:
                        best_score = silhouette_avg
                        full_labels = np.full(len(data), -1)  # Tüm veri seti boyutunda -1 ile doldur
                        full_labels[age_filter] = cluster_labels  # İlgili indekslere küme etiketlerini yerleştir
                        
                        best_config = {
                            'config_name': config_name,
                            'features': feature_set,
                            'scaler': scaler,
                            'method': 'KMeans',
                            'age_group': (age_min, age_max),
                            'clusters': n_clusters
                        }
                        best_labels = full_labels
                except Exception as e:
                    print(f"  Hata: KMeans Age{age_min}-{age_max} (k={n_clusters}): {str(e)}")
    
    # Sonuçları DataFrame'e dönüştür ve kaydet
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "all_clustering_results.csv"), index=False)
    
    # Sonuçları görselleştir
    plt.figure(figsize=(12, 8))
    top_results = results_df.nlargest(20, 'Silhouette')
    
    sns.barplot(x='Silhouette', y='Config', data=top_results)
    plt.title('Top 20 Configurations by Silhouette Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_silhouette_scores.png"), dpi=300)
    plt.close()
    
    # En iyi sonucu görselleştir
    if best_config is not None:
        print(f"\n=== En İyi Kümeleme Konfigürasyonu ===")
        print(f"Konfigürasyon: {best_config['config_name']}")
        print(f"Silhouette Skoru: {best_score:.4f}")
        
        # Kümeleme sonuçlarını kaydet
        result_data = data.copy()
        result_data['BestCluster'] = best_labels
        result_data.to_csv(os.path.join(output_dir, "best_clustering_result.csv"), index=False)
        
        # Küme istatistiklerini hesapla
        # Gürültü noktalarını (-1) hariç tut
        valid_clusters = result_data[result_data['BestCluster'] != -1]
        
        if len(valid_clusters) > 0:
            cluster_stats = valid_clusters.groupby('BestCluster').agg({
                'Age': ['mean', 'count'],
                'Purchase Amount (USD)': ['mean', 'min', 'max'],
                'Review Rating': 'mean',
                'Previous Purchases': 'mean'
            }).round(2)
            
            # İstatistikleri dosyaya kaydet
            with open(os.path.join(output_dir, "best_cluster_stats.txt"), 'w') as f:
                f.write(f"En İyi Kümeleme Konfigürasyonu: {best_config['config_name']}\n")
                f.write(f"Silhouette Skoru: {best_score:.4f}\n\n")
                
                if 'age_group' in best_config:
                    f.write(f"Yaş Grubu: {best_config['age_group'][0]}-{best_config['age_group'][1]}\n")
                
                f.write("Küme İstatistikleri:\n")
                f.write(str(cluster_stats))
                
                # Her küme için kategorik değişkenlerin dağılımını göster
                f.write("\n\nKategorik Değişken Dağılımları:\n")
                categorical_cols = ['Gender', 'Location', 'Season', 'Category', 'Shipping Type', 'Payment Method', 'Frequency of Purchases']
                
                for col in categorical_cols:
                    if col in data.columns:
                        f.write(f"\n{col} Dağılımı:\n")
                        
                        for cluster in sorted(valid_clusters['BestCluster'].unique()):
                            f.write(f"\nKüme {cluster}:\n")
                            cluster_data = valid_clusters[valid_clusters['BestCluster'] == cluster]
                            value_counts = cluster_data[col].value_counts().head(5)
                            f.write(str(value_counts) + "\n")
            
            # Kümeleri görselleştir
            plt.figure(figsize=(10, 6))
            
            # Eğer en iyi konfigürasyon PCA kullandıysa
            if 'pca_components' in best_config and best_config['pca_components'] is not None:
                # PCA'yı yeniden uygula
                X_selected = data[best_config['features']].values
                X_scaled = best_config['scaler'].fit_transform(X_selected)
                
                pca = PCA(n_components=2)  # Görselleştirme için 2 bileşen
                X_pca = pca.fit_transform(X_scaled)
                
                for cluster in sorted(np.unique(best_labels)):
                    if cluster == -1:  # Gürültü noktalarını ayrı göster
                        plt.scatter(X_pca[best_labels == -1, 0], X_pca[best_labels == -1, 1], 
                                    label='Noise', alpha=0.3, color='gray', s=10)
                    else:
                        plt.scatter(X_pca[best_labels == cluster, 0], X_pca[best_labels == cluster, 1], 
                                    label=f'Cluster {cluster}', alpha=0.7)
                
                plt.title(f'PCA Visualization of Best Clustering (Score: {best_score:.4f})')
                plt.xlabel('First Principal Component')
                plt.ylabel('Second Principal Component')
            else:
                # PCA kullanmadıysa, ilk iki özelliği kullan
                features = best_config['features'][:2]  # İlk iki özellik
                
                if len(features) < 2:  # Eğer tek özellik varsa hata oluşmasını önle
                    features = best_config['features']
                    if 'Age' not in features:
                        features = ['Age'] + features
                    features = features[:2]
                
                X_selected = data[features].values
                X_scaled = best_config['scaler'].fit_transform(X_selected)
                
                for cluster in sorted(np.unique(best_labels)):
                    if cluster == -1:  # Gürültü noktalarını ayrı göster
                        plt.scatter(X_scaled[best_labels == -1, 0], X_scaled[best_labels == -1, 1], 
                                   label='Noise', alpha=0.3, color='gray', s=10)
                    else:
                        plt.scatter(X_scaled[best_labels == cluster, 0], X_scaled[best_labels == cluster, 1], 
                                   label=f'Cluster {cluster}', alpha=0.7)
                
                plt.title(f'Best Clustering Visualization (Score: {best_score:.4f})')
                plt.xlabel(features[0])
                plt.ylabel(features[1])
            
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "best_clustering_visualization.png"), dpi=300)
            plt.close()
            
            # Silhouette grafiğini oluştur
            plt.figure(figsize=(10, 8))
            
            # Gürültü noktalarını hariç tut
            valid_indices = best_labels != -1
            
            if np.sum(valid_indices) > 1:
                # Seçilen özellikleri ayır
                X_selected = data[best_config['features']].values[valid_indices]
                X_scaled = best_config['scaler'].fit_transform(X_selected)
                
                # PCA uygulandıysa
                if 'pca_components' in best_config and best_config['pca_components'] is not None:
                    pca = PCA(n_components=best_config['pca_components'])
                    X_for_silhouette = pca.fit_transform(X_scaled)
                else:
                    X_for_silhouette = X_scaled
                
                valid_labels = best_labels[valid_indices]
                
                # Silhouette değerlerini hesapla
                silhouette_vals = silhouette_samples(X_for_silhouette, valid_labels)
                
                # Küme bazında silhouette grafiği
                y_lower = 10
                
                for cluster in sorted(np.unique(valid_labels)):
                    if cluster == -1:
                        continue
                        
                    cluster_silhouette_vals = silhouette_vals[valid_labels == cluster]
                    cluster_silhouette_vals.sort()
                    
                    size_cluster = cluster_silhouette_vals.shape[0]
                    y_upper = y_lower + size_cluster
                    
                    color = plt.cm.viridis(float(cluster) / len(np.unique(valid_labels)))
                    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                                     facecolor=color, edgecolor=color, alpha=0.7)
                    
                    plt.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster))
                    y_lower = y_upper + 10
                
                plt.title(f'Silhouette Plot for Best Clustering (Score: {best_score:.4f})')
                plt.xlabel('Silhouette Coefficient Values')
                plt.ylabel('Cluster Labels')
                plt.axvline(x=best_score, color='red', linestyle='--')
                plt.yticks([])
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "best_silhouette_plot.png"), dpi=300)
                plt.close()
    
    print(f"\nTüm sonuçlar {output_dir} dizinine kaydedildi.")
    
    # 0.5 üzerinde silhouette skoru olan sonuçları göster
    high_scores = results_df[results_df['Silhouette'] >= 0.5]
    if len(high_scores) > 0:
        print("\n=== 0.5 Üzerinde Silhouette Skoru Olan Konfigürasyonlar ===")
        for i, row in high_scores.iterrows():
            print(f"{row['Config']}: {row['Silhouette']:.4f}")
    else:
        print("\n0.5 üzerinde silhouette skoru olan konfigürasyon bulunamadı.")
    
    return results_df, best_config, best_score, best_labels

if __name__ == "__main__":
    # Veriyi analiz et
    results_df, best_config, best_score, best_labels = improve_silhouette('customer_data.csv') 