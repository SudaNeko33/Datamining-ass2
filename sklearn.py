import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
data = df.iloc[:, :-1].values

# # Calculate the distance between two points
def euclidean_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i])**2
    return np.sqrt(sum)

def hierarchical_clustering(data, num_of_clusters):
    # Get the number of instances and features
    num_of_points, num_of_dimensions = data.shape
    # Initialize the clusters as single points
    # clusters为二维数组
    clusters = []
    for instance in range(num_of_points):
        clusters.append([instance])
    # Initialize the distance matrix
    # 初始化距离矩阵为0
    dist_matrix = np.zeros((num_of_points, num_of_points))
    for i in range(num_of_points):
        for j in range(num_of_points):
            if i != j:
                dist_matrix[i, j] = euclidean_distance(data[i], data[j])
    # Merge clusters until desired number of clusters is reached
    # 当聚类数目大于指定的聚类数目时，循环聚类
    while len(clusters) > num_of_clusters:
        # 初始化最小距离为正无穷
        min_dist = np.inf
        # 寻找最近的两个簇，保存在merge_clusters中
        merge_clusters = None
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if i != j:
                    dist = 0
                    # 计算这两个簇的平均距离？
                    for ii in clusters[i]:
                        for jj in clusters[j]:
                            dist += dist_matrix[ii, jj]
                    dist /= len(clusters[i]) * len(clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        # 保存
                        merge_clusters = (i, j)
        # Merge the two closest clusters
        i, j = merge_clusters
        # 将第二个簇的所有实例加入到第一个簇中
        clusters[i] += clusters[j]
        del clusters[j]
        # Update the distance matrix
        for i in range(len(clusters)-1):
            dist = 0
            for ii in clusters[i]:
                for jj in clusters[-1]:
                    dist += dist_matrix[ii, jj]
            dist /= len(clusters[i]) * len(clusters[-1])
            dist_matrix[i, -1] = dist
            dist_matrix[-1, i] = dist
        dist_matrix[-1, -1] = np.inf
    return clusters

# # Cluster the data into 3 clusters
clusters = hierarchical_clustering(data, 3)

# # Plot the clusters
colors = ['r', 'g', 'b']
for i, cluster in enumerate(clusters):
    for instance in cluster:
        plt.scatter(data[instance, 0], data[instance, 1], color=colors[i])
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

# # Compare with the class labels
print('Cluster 1:', df.iloc[clusters[0], -1].value_counts())
print('Cluster 2:', df.iloc[clusters[1], -1].value_counts())
print('Cluster 3:', df.iloc[clusters[2], -1].value_counts())

# # Compare with the class labels
cluster_labels = []
for i, cluster in enumerate(clusters):
    class_counts = df.iloc[cluster, -1].value_counts()
    cluster_label = class_counts.idxmax()
    cluster_labels.extend([cluster_label] * len(cluster))
df['cluster'] = cluster_labels

# # Print the comparison
comparison = df[['class', 'cluster']]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(comparison)