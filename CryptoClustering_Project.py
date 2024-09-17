
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hvplot.pandas

# Step 1: Load and prepare data
df = pd.read_csv('crypto_market_data.csv')

# Step 2: Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop(['coin_id'], axis=1))
scaled_df = pd.DataFrame(scaled_data, columns=df.columns[1:])
scaled_df.index = df['coin_id']

# Step 3: Elbow method to find the best k (Original Data)
inertia = []
k_values = range(1, 12)
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)

# Step 4: Visualizing the Elbow curve
plt.figure(figsize=(8,5))
plt.plot(k_values, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method - Original Data')
plt.show()

# Step 5: K-means clustering using the best k (Assume k=4)
best_k = 4
kmeans = KMeans(n_clusters=best_k)
df['Cluster'] = kmeans.fit_predict(scaled_df)

# Step 6: Visualize the clustering result with hvPlot
scatter_plot_original = df.hvplot.scatter(
    x='price_change_percentage_24h', 
    y='price_change_percentage_7d', 
    by='Cluster', 
    hover_cols=['coin_id']
)
scatter_plot_original.show()

# Step 7: Principal Component Analysis (PCA)
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_df)
explained_variance = pca.explained_variance_ratio_.sum()

# Create DataFrame for PCA data
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2', 'PC3'], index=scaled_df.index)

# Step 8: Elbow method on PCA data
inertia_pca = []
for k in k_values:
    kmeans_pca = KMeans(n_clusters=k)
    kmeans_pca.fit(pca_df)
    inertia_pca.append(kmeans_pca.inertia_)

# Step 9: Visualizing the Elbow curve for PCA
plt.figure(figsize=(8,5))
plt.plot(k_values, inertia_pca, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method - PCA Data')
plt.show()

# Step 10: K-means clustering using PCA data with best k (Assume k=4)
kmeans_pca = KMeans(n_clusters=best_k)
pca_df['Cluster'] = kmeans_pca.fit_predict(pca_df)

# Step 11: Visualize the PCA-based clustering result
scatter_plot_pca = pca_df.hvplot.scatter(
    x='PC1', 
    y='PC2', 
    by='Cluster', 
    hover_cols=['coin_id']
)
scatter_plot_pca.show()
