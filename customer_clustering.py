import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns  

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score  

sns.set_theme(style="whitegrid") 
sns.set_palette("husl")

data = pd.read_csv('customer_data.csv')


features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']


clustering_data = data[features].copy()


scaler = StandardScaler()
data_scaled = scaler.fit_transform(clustering_data)


inertias = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertias.append(kmeans.inertia_)


plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()


n_clusters = 4  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)


fig = plt.figure(figsize=(15, 10))


plt.subplot(2, 2, 1)
scatter = plt.scatter(data['Age'], 
                     data['Purchase Amount (USD)'],
                     c=data['Cluster'],
                     cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Purchase Amount (USD)')
plt.title('Customer Segments: Age vs Purchase Amount')
plt.colorbar(scatter)


plt.subplot(2, 2, 2)
scatter = plt.scatter(data['Previous Purchases'],
                     data['Purchase Amount (USD)'],
                     c=data['Cluster'],
                     cmap='viridis')
plt.xlabel('Previous Purchases')
plt.ylabel('Purchase Amount (USD)')
plt.title('Customer Segments: Previous Purchases vs Purchase Amount')
plt.colorbar(scatter)


plt.subplot(2, 2, 3)
scatter = plt.scatter(data['Review Rating'],
                     data['Purchase Amount (USD)'],
                     c=data['Cluster'],
                     cmap='viridis')
plt.xlabel('Review Rating')
plt.ylabel('Purchase Amount (USD)')
plt.title('Customer Segments: Review Rating vs Purchase Amount')
plt.colorbar(scatter)

plt.tight_layout()
plt.show()


print("\nCluster Statistics:")
cluster_stats = data.groupby('Cluster').agg({
    'Age': ['mean', 'count'],
    'Purchase Amount (USD)': ['mean', 'min', 'max'],
    'Review Rating': 'mean',
    'Previous Purchases': 'mean',
    'Category': lambda x: x.mode().iloc[0],
    'Frequency of Purchases': lambda x: x.mode().iloc[0]
}).round(2)

print(cluster_stats)


print("\nTop Categories by Cluster:")
for cluster in range(n_clusters):
    print(f"\nCluster {cluster}:")
    print(data[data['Cluster'] == cluster]['Category'].value_counts().head(3))

print("\nPurchase Frequency by Cluster:")
for cluster in range(n_clusters):
    print(f"\nCluster {cluster}:")
    print(data[data['Cluster'] == cluster]['Frequency of Purchases'].value_counts().head(3)) 