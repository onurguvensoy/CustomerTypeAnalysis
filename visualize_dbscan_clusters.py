#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization tool for DBSCAN clustering results with high silhouette score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

def load_and_prepare_data(file_path):
    """Load the clustering results and prepare for visualization"""
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    # Filter out noise points for certain visualizations
    valid_data = data[data['BestCluster'] != -1].copy()
    print(f"Found {len(valid_data)} valid cluster points out of {len(data)} total records")
    print(f"Number of clusters: {valid_data['BestCluster'].nunique()}")
    return data, valid_data

def create_output_dir(output_dir="dbscan_visualizations"):
    """Create output directory for visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to {output_dir}/")
    return output_dir

def visualize_age_purchase_relationship(valid_data, output_dir):
    """Create scatter plot of Age vs Purchase Amount colored by cluster"""
    plt.figure(figsize=(14, 10))
    scatter = sns.scatterplot(
        x='Age', 
        y='Purchase Amount (USD)', 
        hue='BestCluster', 
        palette='viridis', 
        data=valid_data,
        s=100,
        alpha=0.7
    )
    plt.title('Customer Segments: Age vs Purchase Amount', fontsize=16)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Purchase Amount (USD)', fontsize=14)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/age_vs_purchase.png", dpi=300)
    plt.close()
    print("✓ Created Age vs Purchase Amount visualization")

def visualize_rating_purchases_relationship(valid_data, output_dir):
    """Create scatter plot of Review Rating vs Previous Purchases colored by cluster"""
    plt.figure(figsize=(14, 10))
    scatter = sns.scatterplot(
        x='Review Rating', 
        y='Previous Purchases', 
        hue='BestCluster', 
        palette='viridis', 
        data=valid_data,
        s=100,
        alpha=0.7
    )
    plt.title('Customer Segments: Review Rating vs Previous Purchases', fontsize=16)
    plt.xlabel('Review Rating', fontsize=14)
    plt.ylabel('Previous Purchases', fontsize=14)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rating_vs_purchases.png", dpi=300)
    plt.close()
    print("✓ Created Review Rating vs Previous Purchases visualization")

def visualize_pca_clusters(data, valid_data, output_dir):
    """Create PCA visualization of the clusters"""
    # Extract the features used for clustering
    features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    X = data[features].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = data['BestCluster'].values
    
    # Get variance explained by each component
    explained_variance = pca.explained_variance_ratio_ * 100
    
    # Create the PCA plot
    plt.figure(figsize=(14, 10))
    
    # Plot noise points first (grey)
    noise_mask = pca_df['Cluster'] == -1
    if noise_mask.sum() > 0:
        sns.scatterplot(
            x='PC1', 
            y='PC2', 
            data=pca_df[noise_mask],
            color='grey',
            alpha=0.3,
            s=50,
            label='Noise'
        )
    
    # Plot actual clusters
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        hue='Cluster', 
        palette='viridis', 
        data=pca_df[~noise_mask],
        s=100,
        alpha=0.7
    )
    
    plt.title('PCA Visualization of Customer Segments (DBSCAN)', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}%)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}%)', fontsize=14)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_visualization.png", dpi=300)
    plt.close()
    print("✓ Created PCA visualization")
    
    # Also create a biplot to understand feature contributions
    plt.figure(figsize=(14, 10))
    
    # Plot the actual data points
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        hue='Cluster', 
        palette='viridis', 
        data=pca_df[pca_df['Cluster'] != -1],
        s=70,
        alpha=0.6
    )
    
    # Plot feature vectors
    feature_vectors = pca.components_.T
    for i, feature in enumerate(features):
        plt.arrow(0, 0, feature_vectors[i, 0]*5, feature_vectors[i, 1]*5, 
                 head_width=0.2, head_length=0.2, fc='red', ec='red')
        plt.text(feature_vectors[i, 0]*5.2, feature_vectors[i, 1]*5.2, feature, 
                color='red', fontsize=12)
    
    plt.title('PCA Biplot: Feature Contributions to Customer Segments', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}%)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_biplot.png", dpi=300)
    plt.close()
    print("✓ Created PCA biplot to show feature contributions")

def visualize_cluster_profiles(valid_data, output_dir):
    """Create parallel coordinates plot for cluster profiles"""
    # Get top clusters with at least 5 members
    cluster_counts = valid_data['BestCluster'].value_counts()
    top_clusters = cluster_counts[cluster_counts >= 5].index.tolist()
    
    # Filter data for top clusters
    top_cluster_data = valid_data[valid_data['BestCluster'].isin(top_clusters)].copy()
    
    # Calculate cluster centers
    cluster_centers = top_cluster_data.groupby('BestCluster')[
        ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    ].mean().reset_index()
    
    # Normalize the features for the plot
    features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    for feature in features:
        min_val = top_cluster_data[feature].min()
        max_val = top_cluster_data[feature].max()
        cluster_centers[f'{feature}_norm'] = (cluster_centers[feature] - min_val) / (max_val - min_val)
    
    # Create parallel coordinates plot
    plt.figure(figsize=(14, 8))
    
    # Setup the color palette
    unique_clusters = cluster_centers['BestCluster'].nunique()
    colors = plt.cm.viridis(np.linspace(0, 1, unique_clusters))
    
    # Create the plot
    for i, (_, row) in enumerate(cluster_centers.iterrows()):
        cluster = int(row['BestCluster'])
        color = colors[i % len(colors)]
        
        plt.plot(features, [row[f'{f}_norm'] for f in features], 
                 c=color, marker='o', markersize=8, label=f"Cluster {cluster}")
    
    plt.xticks(rotation=30, fontsize=12)
    plt.ylabel('Normalized Value', fontsize=14)
    plt.title('Cluster Profiles (Normalized Features)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_profiles.png", dpi=300)
    plt.close()
    print("✓ Created cluster profiles parallel coordinates plot")

def visualize_cluster_heatmap(valid_data, output_dir):
    """Create heatmap of cluster characteristics"""
    # Calculate cluster statistics
    cluster_stats = valid_data.groupby('BestCluster')[
        ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    ].mean()
    
    # Normalize each column for the heatmap
    cluster_stats_norm = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min())
    
    plt.figure(figsize=(12, len(cluster_stats) * 0.4 + 2))
    ax = sns.heatmap(
        cluster_stats_norm.T, 
        annot=cluster_stats.T.round(1),
        fmt='.1f',
        cmap='viridis',
        linewidths=.5,
        cbar_kws={'label': 'Normalized Value'}
    )
    plt.title('Cluster Characteristics Heatmap', fontsize=16)
    plt.ylabel('Feature', fontsize=14)
    plt.xlabel('Cluster ID', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_heatmap.png", dpi=300)
    plt.close()
    print("✓ Created cluster characteristics heatmap")

def visualize_age_distribution(valid_data, output_dir):
    """Create age distribution by cluster type"""
    # Define age groups
    bins = [18, 30, 45, 60, 100]
    labels = ['18-30', '31-45', '46-60', '61+']
    valid_data['Age Group'] = pd.cut(valid_data['Age'], bins=bins, labels=labels, right=False)
    
    # Calculate percentage of each age group in each cluster
    age_cluster_counts = pd.crosstab(
        valid_data['BestCluster'], 
        valid_data['Age Group'], 
        normalize='index'
    ) * 100
    
    # Melt for easier plotting
    age_cluster_melt = age_cluster_counts.reset_index().melt(
        id_vars=['BestCluster'],
        var_name='Age Group',
        value_name='Percentage'
    )
    
    # Find optimal figure size based on number of clusters
    num_clusters = valid_data['BestCluster'].nunique()
    plt.figure(figsize=(14, max(8, num_clusters * 0.3)))
    
    # Create the heatmap
    sns.barplot(
        x='Percentage',
        y='BestCluster',
        hue='Age Group',
        data=age_cluster_melt,
        palette='viridis'
    )
    
    plt.title('Age Group Distribution Across Clusters', fontsize=16)
    plt.xlabel('Percentage (%)', fontsize=14)
    plt.ylabel('Cluster ID', fontsize=14)
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/age_distribution.png", dpi=300)
    plt.close()
    print("✓ Created age distribution visualization")

def visualize_purchase_distribution(valid_data, output_dir):
    """Create purchase amount distribution by cluster"""
    # Define purchase amount groups
    bins = [0, 30, 60, 90, 100]
    labels = ['$0-30', '$31-60', '$61-90', '$91-100']
    valid_data['Purchase Group'] = pd.cut(valid_data['Purchase Amount (USD)'], bins=bins, labels=labels, right=False)
    
    # Calculate percentage of each purchase group in each cluster
    purchase_cluster_counts = pd.crosstab(
        valid_data['BestCluster'], 
        valid_data['Purchase Group'], 
        normalize='index'
    ) * 100
    
    # Melt for easier plotting
    purchase_cluster_melt = purchase_cluster_counts.reset_index().melt(
        id_vars=['BestCluster'],
        var_name='Purchase Group',
        value_name='Percentage'
    )
    
    # Find optimal figure size based on number of clusters
    num_clusters = valid_data['BestCluster'].nunique()
    plt.figure(figsize=(14, max(8, num_clusters * 0.3)))
    
    # Create the barplot
    sns.barplot(
        x='Percentage',
        y='BestCluster',
        hue='Purchase Group',
        data=purchase_cluster_melt,
        palette='viridis'
    )
    
    plt.title('Purchase Amount Distribution Across Clusters', fontsize=16)
    plt.xlabel('Percentage (%)', fontsize=14)
    plt.ylabel('Cluster ID', fontsize=14)
    plt.legend(title='Purchase Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/purchase_distribution.png", dpi=300)
    plt.close()
    print("✓ Created purchase distribution visualization")

def generate_summary_report(valid_data, output_dir):
    """Generate a summary report of cluster characteristics"""
    # Calculate key statistics for each cluster
    cluster_summary = valid_data.groupby('BestCluster').agg({
        'Age': ['mean', 'min', 'max', 'count'],
        'Purchase Amount (USD)': ['mean', 'min', 'max'],
        'Review Rating': ['mean', 'min', 'max'],
        'Previous Purchases': ['mean', 'min', 'max'],
        'Gender': lambda x: x.value_counts().index[0]  # Most common gender
    })
    
    # Format the summary
    cluster_summary.columns = [
        'Age_Mean', 'Age_Min', 'Age_Max', 'Cluster_Size',
        'Purchase_Mean', 'Purchase_Min', 'Purchase_Max',
        'Rating_Mean', 'Rating_Min', 'Rating_Max',
        'PrevPurchases_Mean', 'PrevPurchases_Min', 'PrevPurchases_Max',
        'Dominant_Gender'
    ]
    
    # Save to CSV
    cluster_summary.round(2).reset_index().to_csv(
        f"{output_dir}/cluster_summary.csv", 
        index=False
    )
    print("✓ Generated cluster summary report")
    
    # Create a more detailed text report
    with open(f"{output_dir}/detailed_report.txt", 'w') as f:
        f.write("=== DBSCAN Clustering Results Analysis ===\n\n")
        f.write(f"Total valid clusters: {valid_data['BestCluster'].nunique()}\n")
        f.write(f"Total customers in clusters: {len(valid_data)}\n\n")
        
        f.write("=== Top 10 Largest Clusters ===\n")
        top_clusters = valid_data['BestCluster'].value_counts().head(10)
        for cluster, count in top_clusters.items():
            cluster_data = valid_data[valid_data['BestCluster'] == cluster]
            avg_age = cluster_data['Age'].mean()
            avg_purchase = cluster_data['Purchase Amount (USD)'].mean()
            avg_rating = cluster_data['Review Rating'].mean()
            avg_prev = cluster_data['Previous Purchases'].mean()
            
            f.write(f"\nCluster {cluster} (Size: {count}):\n")
            f.write(f"  Average Age: {avg_age:.1f}\n")
            f.write(f"  Average Purchase: ${avg_purchase:.2f}\n")
            f.write(f"  Average Rating: {avg_rating:.2f}/5.0\n")
            f.write(f"  Average Previous Purchases: {avg_prev:.1f}\n")
        
        f.write("\n=== Customer Segment Profiles ===\n")
        
        # Young segment
        young_clusters = cluster_summary[cluster_summary['Age_Mean'] < 30].index.tolist()
        f.write("\nYoung Customers (Age < 30):\n")
        for cluster in young_clusters:
            cs = cluster_summary.loc[cluster]
            f.write(f"  Cluster {cluster}: Age {cs['Age_Mean']:.1f}, Purchase ${cs['Purchase_Mean']:.2f}, ")
            f.write(f"Rating {cs['Rating_Mean']:.2f}, Previous Purchases {cs['PrevPurchases_Mean']:.1f}\n")
        
        # Middle-aged segment
        mid_clusters = cluster_summary[(cluster_summary['Age_Mean'] >= 30) & 
                                     (cluster_summary['Age_Mean'] < 50)].index.tolist()
        f.write("\nMiddle-Aged Customers (30-49):\n")
        for cluster in mid_clusters:
            cs = cluster_summary.loc[cluster]
            f.write(f"  Cluster {cluster}: Age {cs['Age_Mean']:.1f}, Purchase ${cs['Purchase_Mean']:.2f}, ")
            f.write(f"Rating {cs['Rating_Mean']:.2f}, Previous Purchases {cs['PrevPurchases_Mean']:.1f}\n")
        
        # Senior segment
        senior_clusters = cluster_summary[cluster_summary['Age_Mean'] >= 50].index.tolist()
        f.write("\nSenior Customers (Age 50+):\n")
        for cluster in senior_clusters:
            cs = cluster_summary.loc[cluster]
            f.write(f"  Cluster {cluster}: Age {cs['Age_Mean']:.1f}, Purchase ${cs['Purchase_Mean']:.2f}, ")
            f.write(f"Rating {cs['Rating_Mean']:.2f}, Previous Purchases {cs['PrevPurchases_Mean']:.1f}\n")
    
    print("✓ Generated detailed text report")

def main():
    """Main function to run all visualizations"""
    print("========================================")
    print("DBSCAN Clustering Visualization Tool")
    print("========================================")
    
    # Configuration
    input_file = "improved_clustering_20250425_131958/best_clustering_result.csv"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        print("Please provide the correct path to the clustering results CSV file.")
        return
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Load and prepare data
    data, valid_data = load_and_prepare_data(input_file)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_age_purchase_relationship(valid_data, output_dir)
    visualize_rating_purchases_relationship(valid_data, output_dir)
    visualize_pca_clusters(data, valid_data, output_dir)
    visualize_cluster_profiles(valid_data, output_dir)
    visualize_cluster_heatmap(valid_data, output_dir)
    visualize_age_distribution(valid_data, output_dir)
    visualize_purchase_distribution(valid_data, output_dir)
    
    # Generate summary report
    print("\nGenerating summary reports...")
    generate_summary_report(valid_data, output_dir)
    
    print("\nAll visualizations and reports completed successfully!")
    print(f"Results saved to {output_dir}/ directory")
    print("========================================")

if __name__ == "__main__":
    main() 