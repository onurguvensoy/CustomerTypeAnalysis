import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from datetime import datetime
import warnings
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

scaler = StandardScaler()
for category in merged['Category'].unique():
    print(f'\n===== CATEGORY: {category} =====')
    cat_data = merged[merged['Category'] == category].copy()
    if len(cat_data) < 10:
        print('Skipping due to insufficient samples.')
        continue
    feature_cols = ['Age', 'Quantity', 'Unit Price USD', 'Gender_enc', 'City_enc', 'State_enc', 'Country_enc', 'Continent_enc']
    X = scaler.fit_transform(cat_data[feature_cols])
    best_result = {'score': -1, 'labels': None, 'algo': None, 'params': None}
    # KMeans (Elbow)
    for k in range(2, 8):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sil = None
        try:
            from sklearn.metrics import silhouette_score
            sil = silhouette_score(X, labels)
        except:
            pass
        if sil is not None and sil > best_result['score']:
            best_result = {'score': sil, 'labels': labels, 'algo': 'KMeans', 'params': {'k': k}}
    # DBSCAN
    for eps in [0.3, 0.5, 0.7, 1.0]:
        for min_samples in [5, 10, 15]:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            if len(set(labels)) < 2 or len(set(labels)) == len(labels):
                continue
            try:
                sil = silhouette_score(X, labels)
            except:
                sil = None
            if sil is not None and sil > best_result['score']:
                best_result = {'score': sil, 'labels': labels, 'algo': 'DBSCAN', 'params': {'eps': eps, 'min_samples': min_samples}}
    # Hierarchical
    for n_clusters in range(2, 8):
        for linkage in ['ward', 'complete', 'average']:
            try:
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                labels = model.fit_predict(X)
                if len(set(labels)) < 2:
                    continue
                sil = silhouette_score(X, labels)
                if sil > best_result['score']:
                    best_result = {'score': sil, 'labels': labels, 'algo': 'Hierarchical', 'params': {'n_clusters': n_clusters, 'linkage': linkage}}
            except Exception:
                continue
    # Affinity Propagation
    try:
        model = AffinityPropagation(random_state=42)
        labels = model.fit_predict(X)
        if len(set(labels)) > 1 and len(set(labels)) < len(labels):
            sil = silhouette_score(X, labels)
            if sil > best_result['score']:
                best_result = {'score': sil, 'labels': labels, 'algo': 'AffinityPropagation', 'params': {}}
    except Exception:
        pass
    # Use best clustering
    labels = best_result['labels']
    cat_data['Cluster'] = labels
    print(f"Best clustering: {best_result['algo']} {best_result['params']} (Silhouette: {best_result['score']:.4f})")
    # HEATMAP: Age x Gender
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
    # STACKED BAR: Gender by City
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