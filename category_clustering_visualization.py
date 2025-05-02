import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Load data
sales = pd.read_csv('Sales.csv', encoding='ISO-8859-1')
customers = pd.read_csv('Customers.csv', encoding='ISO-8859-1')
products = pd.read_csv('Products.csv', encoding='ISO-8859-1')

# Preprocess Customers: calculate Age
customers['Birthday'] = pd.to_datetime(customers['Birthday'], errors='coerce')
today = pd.to_datetime('today')
customers['Age'] = customers['Birthday'].apply(lambda x: today.year - x.year if pd.notnull(x) else np.nan)

# Merge Sales with Customers and Products
merged = sales.merge(customers, on='CustomerKey', how='left')
merged = merged.merge(products, on='ProductKey', how='left')
merged = merged.dropna(subset=['Age', 'Gender', 'Category', 'Quantity', 'Order Date'])

# Clean price column
merged['Unit Price USD'] = merged['Unit Price USD'].replace({'[$, ]':''}, regex=True).astype(float)
merged['Order Date'] = pd.to_datetime(merged['Order Date'], errors='coerce')

# Encode categorical variables for clustering
for col in ['Gender', 'City', 'State', 'Country', 'Continent']:
    le = LabelEncoder()
    merged[f'{col}_enc'] = le.fit_transform(merged[col].astype(str))

# General statistics
print('=== GENERAL STATISTICS ===')
print(merged[['Age', 'Quantity', 'Unit Price USD']].describe())
print('\nGender distribution:')
print(merged['Gender'].value_counts())
print('\nCategory distribution:')
print(merged['Category'].value_counts())

# Check for clustering_summary.csv
if not os.path.exists('clustering_summary.csv'):
    print('ERROR: clustering_summary.csv not found. Please run automated_clustering_analysis.py first to generate it.')
    exit(1)

# Load best clustering summary (from previous analysis)
summary = pd.read_csv('clustering_summary.csv')

# For each category, apply best clustering and visualize
scaler = StandardScaler()
for category in merged['Category'].unique():
    print(f'\n===== CATEGORY: {category} =====')
    cat_data = merged[merged['Category'] == category].copy()
    if len(cat_data) < 10:
        print('Skipping due to insufficient samples.')
        continue
    # Features for clustering
    feature_cols = ['Age', 'Quantity', 'Unit Price USD', 'Gender_enc', 'City_enc', 'State_enc', 'Country_enc', 'Continent_enc']
    X = scaler.fit_transform(cat_data[feature_cols])
    # Get best algorithm for this category
    best = summary[summary['Category'] == category].sort_values('Silhouette', ascending=False).iloc[0]
    algo = best['Algorithm']
    print(f'Best clustering: {algo}')
    # Fit clustering
    if 'KMeans' in algo:
        k = int(algo.split('=')[1].split(')')[0])
        model = KMeans(n_clusters=k, random_state=42)
    elif 'DBSCAN' in algo:
        params = algo.split('(')[1].split(')')[0].split(', ')
        eps = float(params[0].split('=')[1])
        min_samples = int(params[1].split('=')[1])
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif 'Hierarchical' in algo:
        n = int(algo.split('=')[2].split(',')[0])
        linkage = algo.split('=')[3].replace(')', '')
        model = AgglomerativeClustering(n_clusters=n, linkage=linkage)
    elif 'AffinityPropagation' in algo:
        model = AffinityPropagation(random_state=42)
    else:
        print('Unknown algorithm, skipping.')
        continue
    labels = model.fit_predict(X)
    cat_data['Cluster'] = labels
    # HEATMAP: Age x Location x Gender (customer count)
    heatmap_data = cat_data.pivot_table(index='Age', columns='Gender', values='CustomerKey', aggfunc='count')
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False)
    plt.title(f'Heatmap: Age x Gender ({category})')
    plt.show()
    # LINE PLOT: Quantity over Order Date
    line_data = cat_data.groupby('Order Date')['Quantity'].sum().sort_index()
    plt.figure(figsize=(10, 4))
    line_data.plot()
    plt.title(f'Quantity Over Time ({category})')
    plt.xlabel('Order Date')
    plt.ylabel('Total Quantity')
    plt.tight_layout()
    plt.show()
    # STACKED BAR: Gender by Location
    stack_data = cat_data.groupby(['City', 'Gender']).size().unstack(fill_value=0)
    stack_data = stack_data[stack_data.sum(axis=1) > 0].head(10)  # Top 10 cities
    stack_data.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='tab20')
    plt.title(f'Stacked Bar: Gender by City ({category})')
    plt.xlabel('City')
    plt.ylabel('Customer Count')
    plt.tight_layout()
    plt.show()
    # PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title(f'PCA Visualization of Clusters ({category})')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show() 