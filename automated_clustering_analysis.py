import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AutomatedClusteringAnalysis:
    def __init__(self, sales_path, customers_path, products_path):
        self.sales_path = sales_path
        self.customers_path = customers_path
        self.products_path = products_path
        self.data = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the data for clustering analysis"""
        # Load the data
        sales = pd.read_csv(self.sales_path, encoding='ISO-8859-1')
        customers = pd.read_csv(self.customers_path, encoding='ISO-8859-1')
        products = pd.read_csv(self.products_path, encoding='ISO-8859-1')
        
        # Preprocess Customers: calculate Age
        today = pd.to_datetime('today')
        customers['Birthday'] = pd.to_datetime(customers['Birthday'], errors='coerce')
        customers['Age'] = customers['Birthday'].apply(lambda x: today.year - x.year if pd.notnull(x) else np.nan)
        
        # Merge Sales with Customers and Products
        merged = sales.merge(customers, on='CustomerKey', how='left')
        merged = merged.merge(products, on='ProductKey', how='left')
        
        # Drop rows with missing essential data
        merged = merged.dropna(subset=['Age', 'Gender', 'Category', 'Quantity'])
        
        # Select features
        features = merged[['Age', 'Gender', 'City', 'State', 'Country', 'Continent', 'Quantity', 'Unit Price USD', 'Category']].copy()
        # Clean price column (remove $ and commas, convert to float)
        features['Unit Price USD'] = features['Unit Price USD'].replace({'[$, ]':''}, regex=True).astype(float)
        
        # Encode categorical variables
        for col in ['Gender', 'City', 'State', 'Country', 'Continent']:
            le = LabelEncoder()
            features[f'{col}_enc'] = le.fit_transform(features[col].astype(str))
        
        # Final feature set for clustering
        feature_cols = ['Age', 'Quantity', 'Unit Price USD', 'Gender_enc', 'City_enc', 'State_enc', 'Country_enc', 'Continent_enc']
        features = features[feature_cols + ['Category']]
        return features
    
    def plot_elbow_method(self, features, category, k_range=range(2, 11)):
        """Plot the Elbow Method for KMeans clustering for a given category"""
        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        plt.figure(figsize=(7, 4))
        plt.plot(list(k_range), inertias, marker='o')
        plt.title(f'Elbow Method for {category}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.xticks(list(k_range))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def evaluate_clustering(self, features, labels, algorithm_name, category):
        """Evaluate clustering results using multiple metrics"""
        if len(set(labels)) > 1:  # Only evaluate if more than one cluster
            silhouette = silhouette_score(features, labels)
            davies_bouldin = davies_bouldin_score(features, labels)
            calinski_harabasz = calinski_harabasz_score(features, labels)
            
            return {
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz,
                'n_clusters': len(set(labels))
            }
        return None
    
    def run_kmeans_analysis(self, features, category, k_range=range(2, 8)):
        """Run K-Means clustering with different k values"""
        print(f"\n=== K-Means Clustering Analysis for Category: {category} ===")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            
            results = self.evaluate_clustering(features, labels, f'K-Means (k={k})', category)
            if results:
                self.results[f'{category} - K-Means (k={k})'] = results
                print(f"K-Means (k={k}):")
                print(f"  Silhouette Score: {results['silhouette_score']:.4f}")
                print(f"  Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")
                print(f"  Calinski-Harabasz Score: {results['calinski_harabasz_score']:.4f}")
    
    def run_dbscan_analysis(self, features, category, eps_range=[0.3, 0.5, 0.7, 1.0], min_samples_range=[5, 10, 15]):
        """Run DBSCAN clustering with different parameters"""
        print(f"\n=== DBSCAN Clustering Analysis for Category: {category} ===")
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(features)
                
                if len(set(labels)) > 1:  # Only evaluate if more than one cluster
                    results = self.evaluate_clustering(features, labels, f'DBSCAN (eps={eps}, min_samples={min_samples})', category)
                    if results:
                        self.results[f'{category} - DBSCAN (eps={eps}, min_samples={min_samples})'] = results
                        print(f"DBSCAN (eps={eps}, min_samples={min_samples}):")
                        print(f"  Silhouette Score: {results['silhouette_score']:.4f}")
                        print(f"  Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")
                        print(f"  Calinski-Harabasz Score: {results['calinski_harabasz_score']:.4f}")
    
    def run_hierarchical_analysis(self, features, category, n_clusters_range=range(2, 8), linkage_methods=['ward', 'complete', 'average']):
        """Run Hierarchical clustering with different parameters"""
        print(f"\n=== Hierarchical Clustering Analysis for Category: {category} ===")
        for n_clusters in n_clusters_range:
            for linkage in linkage_methods:
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                labels = hierarchical.fit_predict(features)
                
                results = self.evaluate_clustering(features, labels, f'Hierarchical (n={n_clusters}, linkage={linkage})', category)
                if results:
                    self.results[f'{category} - Hierarchical (n={n_clusters}, linkage={linkage})'] = results
                    print(f"Hierarchical (n={n_clusters}, linkage={linkage}):")
                    print(f"  Silhouette Score: {results['silhouette_score']:.4f}")
                    print(f"  Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")
                    print(f"  Calinski-Harabasz Score: {results['calinski_harabasz_score']:.4f}")
    
    def find_best_clustering(self, category):
        """Find the best clustering configuration based on combined metrics for a specific category"""
        category_results = {k: v for k, v in self.results.items() if k.startswith(category)}
        if not category_results:
            return None
        
        # Normalize scores
        max_silhouette = max(r['silhouette_score'] for r in category_results.values())
        min_davies = min(r['davies_bouldin_score'] for r in category_results.values())
        max_calinski = max(r['calinski_harabasz_score'] for r in category_results.values())
        
        # Calculate combined score
        for name, result in category_results.items():
            normalized_silhouette = result['silhouette_score'] / max_silhouette
            normalized_davies = min_davies / result['davies_bouldin_score']
            normalized_calinski = result['calinski_harabasz_score'] / max_calinski
            
            # Combined score (equal weights)
            result['combined_score'] = (normalized_silhouette + normalized_davies + normalized_calinski) / 3
        
        # Find best configuration
        best_config = max(category_results.items(), key=lambda x: x[1]['combined_score'])
        return best_config
    
    def visualize_results(self, category):
        """Visualize clustering results for a specific category"""
        category_results = {k: v for k, v in self.results.items() if k.startswith(category)}
        if not category_results:
            return
        
        # Create a DataFrame for visualization
        results_df = pd.DataFrame([
            {
                'Algorithm': name.replace(f'{category} - ', ''),
                'Silhouette Score': result['silhouette_score'],
                'Davies-Bouldin Score': result['davies_bouldin_score'],
                'Calinski-Harabasz Score': result['calinski_harabasz_score'],
                'Number of Clusters': result['n_clusters']
            }
            for name, result in category_results.items()
        ])
        
        # Plot results
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Clustering Results for Category: {category}', fontsize=16)
        
        # Silhouette Score
        plt.subplot(311)
        sns.barplot(x='Algorithm', y='Silhouette Score', data=results_df)
        plt.xticks(rotation=45, ha='right')
        plt.title('Silhouette Scores by Algorithm')
        
        # Davies-Bouldin Score
        plt.subplot(312)
        sns.barplot(x='Algorithm', y='Davies-Bouldin Score', data=results_df)
        plt.xticks(rotation=45, ha='right')
        plt.title('Davies-Bouldin Scores by Algorithm')
        
        # Calinski-Harabasz Score
        plt.subplot(313)
        sns.barplot(x='Algorithm', y='Calinski-Harabasz Score', data=results_df)
        plt.xticks(rotation=45, ha='right')
        plt.title('Calinski-Harabasz Scores by Algorithm')
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self):
        """Run complete clustering analysis for each category"""
        print("Starting automated clustering analysis...")
        
        # Load and prepare data
        features = self.load_and_prepare_data()
        categories = features['Category']
        
        # Get unique categories
        unique_categories = categories.unique()
        
        # Run analysis for each category
        for category in unique_categories:
            print(f"\n{'='*50}")
            print(f"Analyzing Category: {category}")
            print(f"{'='*50}")
            
            # Filter data for current category
            category_mask = categories == category
            category_features = features[category_mask].drop(columns=['Category'])
            
            if len(category_features) < 10:  # Skip categories with too few samples
                print(f"Skipping {category} due to insufficient samples")
                continue
            
            # Elbow Method
            self.plot_elbow_method(category_features, category)
            
            # Run different clustering algorithms
            self.run_kmeans_analysis(category_features, category)
            self.run_dbscan_analysis(category_features, category)
            self.run_hierarchical_analysis(category_features, category)
            
            # Find best clustering for this category
            best_config = self.find_best_clustering(category)
            
            # Visualize results for this category
            self.visualize_results(category)
            
            # Print best configuration for this category
            if best_config:
                print(f"\n=== Best Clustering Configuration for {category} ===")
                print(f"Algorithm: {best_config[0]}")
                print(f"Silhouette Score: {best_config[1]['silhouette_score']:.4f}")
                print(f"Davies-Bouldin Score: {best_config[1]['davies_bouldin_score']:.4f}")
                print(f"Calinski-Harabasz Score: {best_config[1]['calinski_harabasz_score']:.4f}")
                print(f"Number of Clusters: {best_config[1]['n_clusters']}")

# Example usage
if __name__ == "__main__":
    analyzer = AutomatedClusteringAnalysis(
        sales_path="Sales.csv",
        customers_path="Customers.csv",
        products_path="Products.csv"
    )
    analyzer.run_analysis() 