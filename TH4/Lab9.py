import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

set1 = np.random.multivariate_normal([-10, 10], [[1.5, 1], [1, 1.5]], 150)
set2 = np.random.multivariate_normal([-5, 10], [[1, 2], [2, 6]], 150)
set3 = np.random.multivariate_normal([-1, 1], [[4, 0], [0, 4]], 150)
set4 = np.random.multivariate_normal([10, -10], [[4, 0], [0, 4]], 150)
set5 = np.random.multivariate_normal([3, -3], [[4, 0], [0, 4]], 150)

data = np.concatenate((set1, set2, set3, set4, set5))
cluster = np.repeat(np.arange(1, 6), 150)

plt.scatter(data[:, 0], data[:, 1], c=cluster)
plt.title('Generated Data')
plt.show()

def kmeans(data, K=4, stop_crit=1e-3):
    centroids = data[np.random.choice(len(data), K, replace=False)]
    current_stop_crit = 1000
    cluster = np.zeros(len(data))
    converged = False
    it = 1

    while current_stop_crit >= stop_crit and not converged:
        it += 1
        old_centroids = centroids

        for i in range(len(data)):
            min_dist = float('inf')
            for centroid in range(len(centroids)):
                distance_to_centroid = np.sum((centroids[centroid, :] - data[i, :]) ** 2)
                if distance_to_centroid <= min_dist:
                    cluster[i] = centroid
                    min_dist = distance_to_centroid

        for i in range(len(centroids)):
            centroids[i, :] = np.mean(data[cluster == i, :], axis=0)

        current_stop_crit = np.mean((old_centroids - centroids) ** 2)

    return {'data': np.column_stack((data, cluster)), 'centroids': centroids}

result = kmeans(data[:, :2], K=5)
result['data'][:, 2] = result['data'][:, 2].astype(int)
result['centroids'] = np.column_stack((result['centroids'], np.arange(1, 6)))

plt.scatter(result['centroids'][:, 0], result['centroids'][:, 1], c=result['centroids'][:, 2], s=100, alpha=0.7, marker='X')
plt.scatter(result['data'][:, 0], result['data'][:, 1], c=result['data'][:, 2], alpha=0.5)
plt.title('K-means Clustering')
plt.show()
