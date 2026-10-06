# Customer Segmentation using K-Means Clustering

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Customer data
# [Spending, Number of Purchases]
data = [
    [1000, 2],
    [1200, 3],
    [1500, 4],
    [8000, 15],
    [8500, 18],
    [9000, 20],
    [3000, 7],
    [3500, 8],
    [4000, 9]
]

# Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the model
kmeans.fit(data)

# Get cluster numbers
clusters = kmeans.labels_

# Display results
for i in range(len(data)):
    print("Customer", i + 1,
          "Spending =", data[i][0],
          "Purchases =", data[i][1],
          "Cluster =", clusters[i])

# Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Plot the clusters
plt.scatter(
    [x[0] for x in data],
    [x[1] for x in data],
    c=clusters
)

plt.xlabel("Spending")
plt.ylabel("Number of Purchases")
plt.title("Customer Segmentation using K-Means")
plt.show()
