import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
import warnings
import os
import datetime
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

class CustomerSegmentation:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        # Create results directory
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"clustering_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def preprocess_data(self):
        """Preprocess the data for clustering"""
        # Select numerical features
        numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
        X = self.data[numerical_features].copy()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def find_optimal_clusters(self, X_scaled, max_clusters=10):
        """Find optimal number of clusters using multiple methods"""
        # Silhouette scores
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
            davies_scores.append(davies_bouldin_score(X_scaled, cluster_labels))
        
        # Plot the scores
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')
        
        plt.subplot(1, 3, 2)
        plt.plot(range(2, max_clusters + 1), calinski_scores, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Calinski-Harabasz Score')
        plt.title('Calinski-Harabasz Score')
        
        plt.subplot(1, 3, 3)
        plt.plot(range(2, max_clusters + 1), davies_scores, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Davies-Bouldin Score')
        plt.title('Davies-Bouldin Score')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.results_dir, 'optimal_clusters.png'), dpi=300)
        plt.close()
        
        # Save scores to CSV
        scores_df = pd.DataFrame({
            'n_clusters': range(2, max_clusters + 1),
            'silhouette_score': silhouette_scores,
            'calinski_harabasz_score': calinski_scores,
            'davies_bouldin_score': davies_scores
        })
        scores_df.to_csv(os.path.join(self.results_dir, 'cluster_scores.csv'), index=False)
        
        return silhouette_scores, calinski_scores, davies_scores
    
    def apply_clustering_methods(self, X_scaled, n_clusters=4):
        """Apply different clustering methods and compare results"""
        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # Agglomerative
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        agg_labels = agg.fit_predict(X_scaled)
        
        # Store results
        self.data['KMeans_Cluster'] = kmeans_labels
        self.data['DBSCAN_Cluster'] = dbscan_labels
        self.data['Agg_Cluster'] = agg_labels
        
        # Save clustered data
        self.data.to_csv(os.path.join(self.results_dir, 'clustered_data.csv'), index=False)
        
        # Calculate scores
        scores = {
            'KMeans': {
                'Silhouette': silhouette_score(X_scaled, kmeans_labels),
                'Calinski': calinski_harabasz_score(X_scaled, kmeans_labels),
                'Davies': davies_bouldin_score(X_scaled, kmeans_labels)
            },
            'DBSCAN': {
                'Silhouette': silhouette_score(X_scaled, dbscan_labels),
                'Calinski': calinski_harabasz_score(X_scaled, dbscan_labels),
                'Davies': davies_bouldin_score(X_scaled, dbscan_labels)
            },
            'Agglomerative': {
                'Silhouette': silhouette_score(X_scaled, agg_labels),
                'Calinski': calinski_harabasz_score(X_scaled, agg_labels),
                'Davies': davies_bouldin_score(X_scaled, agg_labels)
            }
        }
        
        # Save scores to CSV
        scores_df = pd.DataFrame([
            {'Method': method, 'Metric': metric, 'Score': score}
            for method, method_scores in scores.items()
            for metric, score in method_scores.items()
        ])
        scores_df.to_csv(os.path.join(self.results_dir, 'method_comparison.csv'), index=False)
        
        return scores
    
    def visualize_clusters(self, X_scaled, method='KMeans'):
        """Visualize clusters using PCA"""
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        
        # Fix for the column name mismatch
        if method == 'Agglomerative':
            cluster_column = 'Agg_Cluster'
        else:
            cluster_column = f'{method}_Cluster'
            
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=self.data[cluster_column],
                            cmap='viridis')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title(f'{method} Clustering Results')
        plt.colorbar(scatter)
        
        # Save the figure
        plt.savefig(os.path.join(self.results_dir, f'{method}_clusters.png'), dpi=300)
        plt.close()
    
    def analyze_clusters(self, method='KMeans'):
        """Analyze cluster characteristics"""
        # Fix for the column name mismatch
        if method == 'Agglomerative':
            cluster_col = 'Agg_Cluster'
        else:
            cluster_col = f'{method}_Cluster'
        
        # Basic statistics
        cluster_stats = self.data.groupby(cluster_col).agg({
            'Age': ['mean', 'count'],
            'Purchase Amount (USD)': ['mean', 'min', 'max'],
            'Review Rating': 'mean',
            'Previous Purchases': 'mean',
            'Category': lambda x: x.mode().iloc[0],
            'Frequency of Purchases': lambda x: x.mode().iloc[0]
        }).round(2)
        
        # Save statistics to CSV
        cluster_stats.to_csv(os.path.join(self.results_dir, f'{method}_statistics.csv'))
        
        # Create a text file for detailed analysis
        with open(os.path.join(self.results_dir, f'{method}_analysis.txt'), 'w') as f:
            f.write(f"Cluster Analysis for {method}:\n")
            f.write(str(cluster_stats) + "\n\n")
            
            f.write("Demographic Insights:\n")
            for cluster in sorted(self.data[cluster_col].unique()):
                f.write(f"\nCluster {cluster}:\n")
                
                # Age distribution
                age_stats = self.data[self.data[cluster_col] == cluster]['Age'].describe()
                f.write(f"Age Statistics:\n{age_stats}\n")
                
                # Popular categories
                f.write("\nTop Categories:\n")
                f.write(str(self.data[self.data[cluster_col] == cluster]['Category'].value_counts().head(3)) + "\n")
                
                # Purchase patterns
                f.write("\nPurchase Frequency:\n")
                f.write(str(self.data[self.data[cluster_col] == cluster]['Frequency of Purchases'].value_counts().head(3)) + "\n")
        
        # Create additional visualizations
        self.visualize_cluster_demographics(method)
    
    def visualize_cluster_demographics(self, method):
        """Create additional visualizations for cluster demographics"""
        if method == 'Agglomerative':
            cluster_col = 'Agg_Cluster'
        else:
            cluster_col = f'{method}_Cluster'
        
        # Age distribution by cluster
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=cluster_col, y='Age', data=self.data)
        plt.title(f'Age Distribution by {method} Cluster')
        plt.savefig(os.path.join(self.results_dir, f'{method}_age_distribution.png'), dpi=300)
        plt.close()
        
        # Purchase amount by cluster
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=cluster_col, y='Purchase Amount (USD)', data=self.data)
        plt.title(f'Purchase Amount Distribution by {method} Cluster')
        plt.savefig(os.path.join(self.results_dir, f'{method}_purchase_distribution.png'), dpi=300)
        plt.close()
        
        # Review rating by cluster
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=cluster_col, y='Review Rating', data=self.data)
        plt.title(f'Review Rating Distribution by {method} Cluster')
        plt.savefig(os.path.join(self.results_dir, f'{method}_rating_distribution.png'), dpi=300)
        plt.close()

# Load and process data
data = pd.read_csv('customer_data.csv')
analyzer = CustomerSegmentation(data)

# Preprocess data
X_scaled = analyzer.preprocess_data()

print(f"Saving all results to the folder: {analyzer.results_dir}")

# Find optimal number of clusters
print("Finding optimal number of clusters...")
analyzer.find_optimal_clusters(X_scaled)

# Apply clustering methods
print("\nApplying different clustering methods...")
scores = analyzer.apply_clustering_methods(X_scaled, n_clusters=4)

# Print comparison scores
print("\nClustering Method Comparison:")
comparison_summary = "Clustering Method Comparison:\n\n"
for method, method_scores in scores.items():
    print(f"\n{method}:")
    comparison_summary += f"{method}:\n"
    for metric, score in method_scores.items():
        print(f"{metric}: {score:.4f}")
        comparison_summary += f"{metric}: {score:.4f}\n"
    comparison_summary += "\n"

# Save comparison summary
with open(os.path.join(analyzer.results_dir, 'comparison_summary.txt'), 'w') as f:
    f.write(comparison_summary)

# Visualize clusters for each method
for method in ['KMeans', 'DBSCAN', 'Agglomerative']:
    print(f"\nVisualizing and analyzing {method} clusters...")
    analyzer.visualize_clusters(X_scaled, method=method)
    analyzer.analyze_clusters(method=method)

print(f"\nAnalysis complete! All results saved in the '{analyzer.results_dir}' folder.") 