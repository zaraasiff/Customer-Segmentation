import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import joblib   # for saving models

sns.set(style="whitegrid")

# 🔹 Load dataset
df = pd.read_csv(r"C:\Users\zaraa\Downloads\Mall_Customers.csv")
print("Shape:", df.shape)
print(df.head())

# 🔹 Step 1: Quick Info
print("\nDataset Info")
print(df.describe())
print(df['Gender'].value_counts())

# 🔹 Step 2: Visualization
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution")

plt.subplot(2,2,2)
sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, color='lightgreen')
plt.title("Annual Income Distribution")

plt.subplot(2,2,3)
sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, color='orange')
plt.title("Spending Score Distribution")

plt.subplot(2,2,4)
sns.countplot(x='Gender', data=df, palette=['lightblue','pink'])
plt.title("Gender Distribution")

plt.tight_layout()
plt.show()

# 🔹 Step 3: Relationship Plots
plt.figure(figsize=(12,5))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Age', size='Age', data=df, palette='viridis', alpha=0.7)
plt.title("Income vs Spending Score (colored by Age)")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Spending Score (1-100)', data=df, color='purple')
plt.title("Age vs Spending Score")
plt.show()

# 🔹 Preprocessing
df2 = df.copy()
df2 = df2.dropna(subset=['Age','Gender'])
df2['Gender_code'] = pd.get_dummies(df2['Gender'], drop_first=True)
features2 = ['Age','Annual Income (k$)','Spending Score (1-100)','Gender_code']
X2 = df2[features2]
X2_scaled = StandardScaler().fit_transform(X2)

# Select features for clustering (only income & score)
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled shape:", X_scaled.shape)

# 🔹 Elbow Method
wcss = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(7,4))
plt.plot(K_range, wcss, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# 🔹 Silhouette Score
sil_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)

plt.figure(figsize=(7,4))
plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.show()
print("Best k by silhouette:", K_range[np.argmax(sil_scores)])

# 🔹 Final KMeans
final_k = 5
kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("\nCluster Counts:\n", df['Cluster'].value_counts().sort_index())

# Cluster centers
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers_original, columns=features)
print("\nCluster Centers:\n", centers_df)

# Visualize clusters
plt.figure(figsize=(8,6))
palette = sns.color_palette("viridis", final_k)
sns.scatterplot(x=X[features[0]], y=X[features[1]],
                hue=df['Cluster'], palette=palette, s=80, legend='full')
plt.scatter(centers_original[:,0], centers_original[:,1],
            s=300, c='red', marker='X', label='Centroids', edgecolor='k')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title("KMeans Clusters")
plt.legend(title='Cluster')
plt.show()

# 🔹 DBSCAN
nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)
distances = np.sort(distances[:,4])
plt.figure(figsize=(7,4))
plt.plot(distances)
plt.title("k-distance Graph (for eps)")
plt.show()

db = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = db.fit_predict(X_scaled)
print("\nDBSCAN Cluster Count: ")
