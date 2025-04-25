import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import os
import warnings
warnings.filterwarnings('ignore')

def improve_silhouette_scores(data_file, output_dir='improved_results'):
    """
    Improve silhouette scores by exploring different parameters.
    
    Parameters:
    -----------
    data_file : str
        Path to customer data CSV file
    output_dir : str
        Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = pd.read_csv(data_file)
    print(f"Loaded data with {data.shape[0]} customers and {data.shape[1]} features")
    
    # Define features for clustering
    features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    
    # 1. Try different scalers
    print("\n1. Testing different scalers...")
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    scaler_results = []
    X = data[features].values
    
    for name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            try:
                sil_score = silhouette_score(X_scaled, labels)
                scaler_results.append({
                    'Scaler': name,
                    'Clusters': n_clusters,
                    'Silhouette': sil_score
                })
                print(f"  {name} with {n_clusters} clusters: {sil_score:.4f}")
            except:
                print(f"  Error calculating silhouette for {name} with {n_clusters} clusters")
    
    # Visualize scaler results
    scaler_df = pd.DataFrame(scaler_results)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Clusters', y='Silhouette', hue='Scaler', data=scaler_df, marker='o')
    plt.title('Silhouette Scores by Scaler and Cluster Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaler_comparison.png'), dpi=300)
    plt.close()
    
    # Find best scaler
    best_scaler_row = scaler_df.loc[scaler_df['Silhouette'].idxmax()]
    best_scaler_name = best_scaler_row['Scaler']
    best_n_clusters = int(best_scaler_row['Clusters'])
    best_scaler = scalers[best_scaler_name]
    
    print(f"\nBest scaler: {best_scaler_name} with {best_n_clusters} clusters")
    print(f"Best silhouette score: {best_scaler_row['Silhouette']:.4f}")
    
    # 2. Try using categorical information as pre-labels
    print("\n2. Using categorical information as guidance...")
    categorical_cols = ['Category', 'Frequency of Purchases']
    categorical_results = []
    
    # Scale data with best scaler
    X_scaled = best_scaler.fit_transform(X)
    
    # Baseline KMeans without guidance
    kmeans_baseline = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    baseline_labels = kmeans_baseline.fit_predict(X_scaled)
    baseline_score = silhouette_score(X_scaled, baseline_labels)
    categorical_results.append({
        'Method': 'No Guidance (Baseline)',
        'Silhouette': baseline_score
    })
    print(f"  Baseline (no guidance): {baseline_score:.4f}")
    
    # Try using each categorical column as guidance
    for cat_col in categorical_cols:
        if cat_col in data.columns:
            # Get unique categories
            categories = data[cat_col].unique()
            n_categories = len(categories)
            
            if n_categories < 2:
                print(f"  Skipping {cat_col} - not enough unique values")
                continue
            
            # Calculate initial centroids based on categories
            initial_centroids = np.zeros((best_n_clusters, len(features)))
            
            # If too few categories, use them and add random ones
            if n_categories < best_n_clusters:
                # Use available categories
                for i, category in enumerate(categories):
                    category_data = X_scaled[data[cat_col] == category]
                    if len(category_data) > 0:
                        initial_centroids[i] = category_data.mean(axis=0)
                
                # Use k-means++ for remaining centroids
                kmeans_guided = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            
            # If too many categories, merge similar ones
            elif n_categories > best_n_clusters:
                # Create centroids for each category
                category_centroids = np.zeros((n_categories, len(features)))
                for i, category in enumerate(categories):
                    category_data = X_scaled[data[cat_col] == category]
                    if len(category_data) > 0:
                        category_centroids[i] = category_data.mean(axis=0)
                
                # Cluster the category centroids
                meta_kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
                meta_kmeans.fit(category_centroids)
                
                # Use these as initial centroids
                initial_centroids = meta_kmeans.cluster_centers_
                kmeans_guided = KMeans(n_clusters=best_n_clusters, init=initial_centroids, n_init=1, random_state=42)
            
            # If exact number of categories
            else:
                for i, category in enumerate(categories):
                    category_data = X_scaled[data[cat_col] == category]
                    if len(category_data) > 0:
                        initial_centroids[i] = category_data.mean(axis=0)
                
                kmeans_guided = KMeans(n_clusters=best_n_clusters, init=initial_centroids, n_init=1, random_state=42)
            
            # Fit and evaluate
            guided_labels = kmeans_guided.fit_predict(X_scaled)
            guided_score = silhouette_score(X_scaled, guided_labels)
            
            categorical_results.append({
                'Method': f'Guided by {cat_col}',
                'Silhouette': guided_score
            })
            print(f"  Guided by {cat_col}: {guided_score:.4f}")
            
            # Save the guided clusters to the data
            data[f'Guided_{cat_col}_Cluster'] = guided_labels
    
    # Visualize categorical guidance results
    cat_df = pd.DataFrame(categorical_results)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Method', y='Silhouette', data=cat_df)
    plt.title('Silhouette Scores With Different Guidance Methods')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'categorical_guidance.png'), dpi=300)
    plt.close()
    
    # 3. Try dimension reduction with PCA
    print("\n3. Testing PCA dimension reduction...")
    pca_results = []
    
    for n_components in range(2, len(features) + 1):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        explained_var = sum(pca.explained_variance_ratio_) * 100
        
        kmeans_pca = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        pca_labels = kmeans_pca.fit_predict(X_pca)
        
        try:
            pca_score = silhouette_score(X_pca, pca_labels)
            pca_results.append({
                'Components': n_components,
                'Explained_Variance': explained_var,
                'Silhouette': pca_score
            })
            print(f"  PCA with {n_components} components ({explained_var:.2f}% variance): {pca_score:.4f}")
        except:
            print(f"  Error with PCA using {n_components} components")
    
    # Visualize PCA results
    pca_df = pd.DataFrame(pca_results)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Components', y='Silhouette', data=pca_df, marker='o', label='Silhouette')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    sns.lineplot(x='Components', y='Explained_Variance', data=pca_df, marker='s', color='r', ax=ax2, label='Variance')
    ax1.set_xlabel('PCA Components')
    ax1.set_ylabel('Silhouette Score')
    ax2.set_ylabel('Explained Variance (%)')
    plt.title('PCA Components vs Silhouette Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_analysis.png'), dpi=300)
    plt.close()
    
    # Find best PCA configuration
    if len(pca_df) > 0:
        best_pca_row = pca_df.loc[pca_df['Silhouette'].idxmax()]
        best_components = int(best_pca_row['Components'])
        best_pca_score = best_pca_row['Silhouette']
        
        # Apply best PCA
        best_pca = PCA(n_components=best_components)
        X_best_pca = best_pca.fit_transform(X_scaled)
        
        # Apply K-means with best configuration
        kmeans_best_pca = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        best_pca_labels = kmeans_best_pca.fit_predict(X_best_pca)
        
        # Save the PCA clusters
        data['PCA_Optimized_Cluster'] = best_pca_labels
        
        print(f"\nBest PCA configuration: {best_components} components")
        print(f"PCA Silhouette score: {best_pca_score:.4f}")
    
    # Save the final improved dataset with all cluster labels
    data.to_csv(os.path.join(output_dir, 'improved_clusters.csv'), index=False)
    
    print(f"\nImproved clustering results saved to {output_dir}/improved_clusters.csv")
    print("Summary of silhouette score improvements:")
    print(f"1. Best scaler ({best_scaler_name}): {best_scaler_row['Silhouette']:.4f}")
    
    best_categorical = max(categorical_results, key=lambda x: x['Silhouette'])
    print(f"2. Best categorical guidance ({best_categorical['Method']}): {best_categorical['Silhouette']:.4f}")
    
    if len(pca_df) > 0:
        print(f"3. Best PCA configuration ({best_components} components): {best_pca_score:.4f}")
    
    return data

if __name__ == "__main__":
    improve_silhouette_scores("customer_data.csv") 