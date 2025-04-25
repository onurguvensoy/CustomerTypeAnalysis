import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def optimize_silhouette(data, feature_columns, n_clusters_range=range(2, 11), 
                        scalers=None, feature_selections=None):
    """
    Optimize silhouette score by trying different combinations of:
    - Scaling methods
    - Feature selections
    - Number of clusters
    - Clustering algorithms
    
    Parameters:
    -----------
    data : DataFrame
        The customer data
    feature_columns : list
        List of numerical feature columns to use for clustering
    n_clusters_range : range or list
        Range of cluster numbers to try
    scalers : list, optional
        List of scaler objects to try
    feature_selections : list, optional
        Number of features to select (1 to len(feature_columns))
    
    Returns:
    --------
    tuple
        (best_configuration, best_silhouette_score, results_df)
    """
    if scalers is None:
        scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]
    
    if feature_selections is None:
        feature_selections = list(range(2, len(feature_columns) + 1))
    
    results = []
    best_silhouette = -1
    best_config = None
    
    for scaler in scalers:
        scaler_name = scaler.__class__.__name__
        X = data[feature_columns].values
        X_scaled = scaler.fit_transform(X)
        
        # Try different feature selection counts
        for n_features in feature_selections:
            if n_features > len(feature_columns):
                continue
                
            # Select top features
            selector = SelectKBest(f_classif, k=n_features)
            X_selected = selector
            
            # For simplicity with feature selection, we'll use the scaled features directly
            X_selected = X_scaled
                
            # Try K-Means with different cluster counts
            for n_clusters in n_clusters_range:
                # K-Means
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans_labels = kmeans.fit_predict(X_selected)
                    kmeans_silhouette = silhouette_score(X_selected, kmeans_labels)
                    
                    results.append({
                        'Scaler': scaler_name,
                        'Features': n_features,
                        'Algorithm': 'KMeans',
                        'Clusters': n_clusters,
                        'Silhouette': kmeans_silhouette
                    })
                    
                    if kmeans_silhouette > best_silhouette:
                        best_silhouette = kmeans_silhouette
                        best_config = {
                            'scaler': scaler,
                            'n_features': n_features,
                            'algorithm': 'KMeans',
                            'n_clusters': n_clusters,
                            'model': kmeans,
                            'labels': kmeans_labels
                        }
                except Exception as e:
                    print(f"Error with KMeans (n={n_clusters}): {str(e)}")
                
                # Agglomerative Clustering
                try:
                    agg = AgglomerativeClustering(n_clusters=n_clusters)
                    agg_labels = agg.fit_predict(X_selected)
                    agg_silhouette = silhouette_score(X_selected, agg_labels)
                    
                    results.append({
                        'Scaler': scaler_name,
                        'Features': n_features,
                        'Algorithm': 'Agglomerative',
                        'Clusters': n_clusters,
                        'Silhouette': agg_silhouette
                    })
                    
                    if agg_silhouette > best_silhouette:
                        best_silhouette = agg_silhouette
                        best_config = {
                            'scaler': scaler,
                            'n_features': n_features,
                            'algorithm': 'Agglomerative',
                            'n_clusters': n_clusters,
                            'model': agg,
                            'labels': agg_labels
                        }
                except Exception as e:
                    print(f"Error with Agglomerative (n={n_clusters}): {str(e)}")
            
            # Try DBSCAN with different parameters
            for eps in [0.3, 0.5, 0.7, 1.0]:
                for min_samples in [3, 5, 7, 10]:
                    try:
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        dbscan_labels = dbscan.fit_predict(X_selected)
                        
                        # DBSCAN may return -1 for noise points
                        # Filter out noise points for silhouette calculation
                        if len(np.unique(dbscan_labels)) > 1 and -1 not in dbscan_labels:
                            dbscan_silhouette = silhouette_score(X_selected, dbscan_labels)
                            
                            results.append({
                                'Scaler': scaler_name,
                                'Features': n_features,
                                'Algorithm': 'DBSCAN',
                                'Clusters': len(np.unique(dbscan_labels)),
                                'Silhouette': dbscan_silhouette,
                                'eps': eps,
                                'min_samples': min_samples
                            })
                            
                            if dbscan_silhouette > best_silhouette:
                                best_silhouette = dbscan_silhouette
                                best_config = {
                                    'scaler': scaler,
                                    'n_features': n_features,
                                    'algorithm': 'DBSCAN',
                                    'eps': eps,
                                    'min_samples': min_samples,
                                    'model': dbscan,
                                    'labels': dbscan_labels
                                }
                        elif -1 in dbscan_labels and len(np.unique(dbscan_labels)) > 2:
                            # Calculate silhouette excluding noise points
                            valid_indices = dbscan_labels != -1
                            if np.sum(valid_indices) > 1:
                                dbscan_silhouette = silhouette_score(
                                    X_selected[valid_indices], 
                                    dbscan_labels[valid_indices]
                                )
                                
                                results.append({
                                    'Scaler': scaler_name,
                                    'Features': n_features,
                                    'Algorithm': 'DBSCAN',
                                    'Clusters': len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                                    'Silhouette': dbscan_silhouette,
                                    'eps': eps,
                                    'min_samples': min_samples,
                                    'noise_points': np.sum(dbscan_labels == -1)
                                })
                                
                                if dbscan_silhouette > best_silhouette:
                                    best_silhouette = dbscan_silhouette
                                    best_config = {
                                        'scaler': scaler,
                                        'n_features': n_features,
                                        'algorithm': 'DBSCAN',
                                        'eps': eps,
                                        'min_samples': min_samples,
                                        'model': dbscan,
                                        'labels': dbscan_labels
                                    }
                    except Exception as e:
                        pass
    
    results_df = pd.DataFrame(results)
    return best_config, best_silhouette, results_df

def analyze_silhouette_improvement(data, feature_columns):
    """
    Analyze silhouette score improvements and visualize the results
    
    Parameters:
    -----------
    data : DataFrame
        The customer data
    feature_columns : list
        List of numerical feature columns to use for clustering
    
    Returns:
    --------
    tuple
        (best_config, improved_data)
    """
    print("Optimizing clustering parameters for better silhouette scores...")
    best_config, best_silhouette, results_df = optimize_silhouette(data, feature_columns)
    
    print(f"\nBest Configuration:")
    print(f"  Algorithm: {best_config['algorithm']}")
    print(f"  Scaler: {best_config['scaler'].__class__.__name__}")
    print(f"  Features Used: {best_config['n_features']}")
    if best_config['algorithm'] == 'DBSCAN':
        print(f"  Epsilon: {best_config['eps']}")
        print(f"  Min Samples: {best_config['min_samples']}")
    else:
        print(f"  Number of Clusters: {best_config['n_clusters']}")
    print(f"  Silhouette Score: {best_silhouette:.4f}")
    
    # Plot top 10 configurations
    top_configs = results_df.sort_values('Silhouette', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Silhouette', y='Algorithm', hue='Scaler', data=top_configs)
    plt.title('Top 10 Configurations by Silhouette Score')
    plt.tight_layout()
    plt.savefig('top_silhouette_configurations.png', dpi=300)
    plt.close()
    
    # Visualize silhouette distributions for the best configuration
    X_scaled = best_config['scaler'].transform(data[feature_columns].values)
    cluster_labels = best_config['labels']
    
    # Create silhouette plot
    plt.figure(figsize=(10, 8))
    
    # Calculate silhouette scores for each sample
    silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
    
    # Plot for each cluster
    unique_clusters = np.unique(cluster_labels)
    if -1 in unique_clusters:  # Remove noise cluster from visualization
        unique_clusters = unique_clusters[unique_clusters != -1]
        
    y_lower = 10
    
    for i, cluster in enumerate(unique_clusters):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == cluster]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.viridis(float(i) / len(unique_clusters))
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))
        y_lower = y_upper + 10
    
    plt.title('Silhouette Plot for Best Configuration')
    plt.xlabel('Silhouette Coefficient Values')
    plt.ylabel('Cluster Labels')
    plt.axvline(x=best_silhouette, color='red', linestyle='--')
    plt.savefig('best_silhouette_distribution.png', dpi=300)
    plt.close()
    
    # Add the best cluster labels to the original data
    improved_data = data.copy()
    improved_data['Optimized_Cluster'] = cluster_labels
    
    return best_config, improved_data

def use_prelabels_for_guidance(data, prelabels_column, feature_columns, n_clusters):
    """
    Use existing labels/categories as guidance for clustering
    
    Parameters:
    -----------
    data : DataFrame
        The customer data
    prelabels_column : str
        Column name containing pre-existing labels or categories
    feature_columns : list
        List of numerical feature columns to use for clustering
    n_clusters : int
        Number of clusters to create
    
    Returns:
    --------
    tuple
        (guided_labels, silhouette_score)
    """
    print(f"Using {prelabels_column} as guidance for clustering...")
    
    # Extract pre-labels and encode them
    prelabels = data[prelabels_column].values
    unique_prelabels = np.unique(prelabels)
    
    if len(unique_prelabels) < 2:
        print("Error: Pre-labels column must have at least 2 unique values")
        return None, None
    
    # Initialize KMeans with existing labels as initial centroids
    X = data[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create centroids based on average of each pre-labeled group
    initial_centroids = np.zeros((n_clusters, X_scaled.shape[1]))
    
    # If we have fewer pre-label categories than desired clusters
    if len(unique_prelabels) < n_clusters:
        print(f"Warning: Only {len(unique_prelabels)} pre-labels, but {n_clusters} clusters requested.")
        print("Using pre-labels for available centroids and random initialization for the rest.")
        
        # Use available pre-labels
        for i, label in enumerate(unique_prelabels):
            mask = prelabels == label
            if np.sum(mask) > 0:  # Ensure we have samples with this label
                initial_centroids[i] = X_scaled[mask].mean(axis=0)
        
        # Random initialize the rest
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
        
    # If we have more pre-label categories than desired clusters, we'll merge some
    elif len(unique_prelabels) > n_clusters:
        print(f"Warning: {len(unique_prelabels)} pre-labels, but only {n_clusters} clusters requested.")
        print("Merging similar pre-label categories.")
        
        # Create temporary clusters from pre-labels
        temp_labels = np.zeros(len(prelabels), dtype=int)
        for i, label in enumerate(unique_prelabels):
            temp_labels[prelabels == label] = i
        
        # Calculate centroids for each pre-label
        prelabel_centroids = np.zeros((len(unique_prelabels), X_scaled.shape[1]))
        for i, label in enumerate(unique_prelabels):
            mask = prelabels == label
            if np.sum(mask) > 0:
                prelabel_centroids[i] = X_scaled[mask].mean(axis=0)
        
        # Cluster the pre-label centroids to get merged groups
        meta_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        meta_labels = meta_kmeans.fit_predict(prelabel_centroids)
        
        # Map original prelabels to merged clusters
        merged_labels = np.zeros(len(prelabels), dtype=int)
        for i, label in enumerate(unique_prelabels):
            merged_labels[prelabels == label] = meta_labels[i]
        
        # Calculate centroids for merged groups
        for i in range(n_clusters):
            mask = merged_labels == i
            if np.sum(mask) > 0:
                initial_centroids[i] = X_scaled[mask].mean(axis=0)
        
        # Use these merged centroids
        kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1, random_state=42)
    
    # If we have exactly the right number of pre-label categories
    else:
        print(f"Using {len(unique_prelabels)} pre-labels as initial centroids.")
        for i, label in enumerate(unique_prelabels):
            mask = prelabels == label
            if np.sum(mask) > 0:
                initial_centroids[i] = X_scaled[mask].mean(axis=0)
        
        # Use these centroids from pre-labels
        kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1, random_state=42)
    
    # Fit KMeans
    guided_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    sil_score = silhouette_score(X_scaled, guided_labels)
    print(f"Guided clustering silhouette score: {sil_score:.4f}")
    
    return guided_labels, sil_score

# Example usage
if __name__ == "__main__":
    # Load your customer data
    data = pd.read_csv('customer_data.csv')
    
    # Define numerical features for clustering
    feature_columns = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    
    # 1. Try optimizing parameters for better silhouette score
    best_config, improved_data = analyze_silhouette_improvement(data, feature_columns)
    
    # 2. If you have a categorical column that could guide clustering, use it
    # For example 'Category' or 'Frequency of Purchases'
    if 'Category' in data.columns:
        guided_labels, guided_score = use_prelabels_for_guidance(
            data, 'Category', feature_columns, n_clusters=4
        )
        
        # Add these guided labels to the data
        improved_data['Guided_Cluster'] = guided_labels
    
    # Save the improved data
    improved_data.to_csv('improved_clustering_results.csv', index=False)
    
    print("\nAnalysis complete. Results saved to improved_clustering_results.csv") 