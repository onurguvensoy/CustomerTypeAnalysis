import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Category to cluster count mapping
category_k = {
    'Cameras and camcorders': 4,
    'Home Appliances': 3,
    'Computers': 3,
    'TV and Video': 3,
    'Cell phones': 4,
    'Music, Movies and Audio Books': 4,
    'Games and Toys': 4,
    'Audio': 4
}

# Load data
sales = pd.read_csv('Sales.csv', encoding='ISO-8859-1')
customers = pd.read_csv('Customers.csv', encoding='ISO-8859-1')
products = pd.read_csv('Products.csv', encoding='ISO-8859-1')

# Preprocess Customers: calculate Age and AgeGroup
customers['Birthday'] = pd.to_datetime(customers['Birthday'], errors='coerce')
today = pd.to_datetime('today')
customers['Age'] = customers['Birthday'].apply(lambda x: today.year - x.year if pd.notnull(x) else np.nan)
bins = [0, 18, 25, 35, 45, 55, 65, 100]
labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']
customers['AgeGroup'] = pd.cut(customers['Age'], bins=bins, labels=labels, right=False)

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
def print_general_stats(df):
    print('=== GENERAL STATISTICS ===')
    print(df[['Age', 'Quantity', 'Unit Price USD']].describe())
    print('\nGender distribution:')
    print(df['Gender'].value_counts())
    print('\nCategory distribution:')
    print(df['Category'].value_counts())

# PCA function
def perform_pca(data, scaled_features, cluster_column):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    cluster_labels = data[cluster_column].reset_index(drop=True)
    final_df = pd.concat([principal_df, cluster_labels], axis=1)
    return final_df

# KMeans clustering visualization
def visualize_kmeans_clusters(pca_df, cluster_column, category, save_folder):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue=cluster_column, data=pca_df, palette='tab10', legend='full')
    plt.title(f'KMeans PCA Plot: {category}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend(title='Cluster', loc='best')
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f'{save_folder}/kmeans_pca_plot.png')
    plt.close()

print_general_stats(merged)

scaler = StandardScaler()
for category, k in category_k.items():
    print(f'\n===== CATEGORY: {category} (KMeans k={k}) =====')
    cat_data = merged[merged['Category'] == category].copy()
    if len(cat_data) < 10:
        print('Skipping due to insufficient samples.')
        continue
    # AgeGroup for this category
    cat_data['AgeGroup'] = pd.cut(cat_data['Age'], bins=bins, labels=labels, right=False)
    feature_cols = ['Age', 'Quantity', 'Unit Price USD', 'Gender_enc', 'City_enc', 'State_enc', 'Country_enc', 'Continent_enc']
    X = scaler.fit_transform(cat_data[feature_cols])
    model = KMeans(n_clusters=k, random_state=42)
    labels_ = model.fit_predict(X)
    cat_data['Cluster'] = labels_
    # --- STATISTICAL TABLES ---
    print('\nCluster Sizes:')
    print(cat_data['Cluster'].value_counts().sort_index())
    print('\nCluster Means:')
    print(cat_data.groupby('Cluster')[['Age', 'Quantity', 'Unit Price USD']].mean())
    print('\nGender Distribution per Cluster:')
    print(pd.crosstab(cat_data['Cluster'], cat_data['Gender'], normalize='index').round(2))
    print('\nAge Group Distribution per Cluster:')
    print(pd.crosstab(cat_data['Cluster'], cat_data['AgeGroup'], normalize='index').round(2))
    # --- HEATMAP: AgeGroup x Gender, faceted by top 3 countries ---
    top_countries = cat_data['Country'].value_counts().head(3).index
    for country in top_countries:
        subset = cat_data[cat_data['Country'] == country]
        heatmap_data = subset.pivot_table(index='AgeGroup', columns='Gender', values='CustomerKey', aggfunc='count', fill_value=0)
        plt.figure(figsize=(6, 4))
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d')
        plt.title(f'Heatmap: AgeGroup x Gender ({category}, {country})')
        plt.ylabel('Age Group')
        plt.xlabel('Gender')
        plt.tight_layout()
        folder = f'./{category}/KMeans'
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/heatmap_{country}.png')
        plt.close()
    # --- LINE PLOT: Quantity over Order Date (with rolling mean, by cluster) ---
    plt.figure(figsize=(12, 5))
    for cluster in sorted(cat_data['Cluster'].unique()):
        cluster_data = cat_data[cat_data['Cluster'] == cluster]
        line = cluster_data.groupby('Order Date')['Quantity'].sum().sort_index().rolling(7, min_periods=1).mean()
        plt.plot(line.index, line.values, label=f'Cluster {cluster}')
    plt.title(f'Quantity Over Time by Cluster ({category})')
    plt.xlabel('Order Date')
    plt.ylabel('Rolling Mean Quantity (7d)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    folder = f'./{category}/KMeans'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/quantity_over_time.png')
    plt.close()
    # --- STACKED BAR: AgeGroup x Gender by top 5 countries ---
    stack_data = cat_data[cat_data['Country'].isin(cat_data['Country'].value_counts().head(5).index)]
    stack = stack_data.groupby(['Country', 'AgeGroup', 'Gender']).size().unstack(fill_value=0)
    stack = stack.groupby(['Country', 'AgeGroup']).sum().unstack(fill_value=0)
    stack.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
    plt.title(f'Stacked Bar: AgeGroup x Gender by Country ({category})')
    plt.xlabel('Country, Age Group')
    plt.ylabel('Customer Count')
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.savefig(f'{folder}/stacked_bar.png')
    plt.close()
    # --- PCA Visualization ---
    pca_df = perform_pca(cat_data, X, 'Cluster')
    visualize_kmeans_clusters(pca_df, 'Cluster', category, folder)
    # Save segmented data
    cat_data.to_csv(f'{folder}/segmented_data.csv', index=False) 