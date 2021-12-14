from typing import List

import numpy
import numpy as np
from clustering_utils import (
    generate_data,
    N_CLUSTERS,
    evaluate,
    ClusteredDataset,
    Embedding,
    euclidean_distance,
    calculate_mean,
)


class ClusteringModel:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.centroids = None # to hold centroids of the clusters

    def __pick_cluster_centers(self, data):
        """
        Takes input data and pick random centroids to use when initializing the training
        Input:
            data: the dataset to cluster from
        Returns:
            centroids: a numpy array of shape n_clusters
        """
        centroids = np.random.randint(0, data.shape[0], size=self.n_clusters)

        return data[centroids]

    def __assign_cluster_labels(self, data, centroids):
        """
        updates the cluster labels using latest centroids.
        Input:
            data: the data to cluster from
            centroids: the centroids returned as part of the update centroids method
        """
        data_to_centroid_diff = data[:, None] - centroids[None]  # (n, m, d)
        distances = np.einsum('nmd,nmd->nm', data_to_centroid_diff, data_to_centroid_diff)  # (n, m)
        cluster_labels = np.argmin(distances, axis=1)  # (n,)
        return cluster_labels

    def __update_centroids(self, data, cluster_labels):
        """
        update the centroids using the new cluster labels.
        Input:
            data: the data to cluster from
            cluster_labels: the new cluster labels
        """
        centroids = []
        for c in range(self.n_clusters):
            data_c = data[np.where(cluster_labels == c)[0]]
            centroid = data_c.mean(axis=0)
            centroids.append(centroid)
        return np.array(centroids)

    def __euclidean_distance(self, data, centroids):
        """
        Semi vectorized method to calculate euclidean distance of each entry of the data to centroids
        Input:
            data: the data to cluster from
            centroids: the current centroids
        """
        return np.linalg.norm(data - centroids, axis=1) ** 2

    def fit(self, dataset: List[Embedding], epochs_to_train: int) -> ClusteredDataset:
        """
        Train a clustering model on the provided data.
        Returns a dict mapping cluster IDs to a list of embeddings in that cluster.
        """
        # I need to use numpy for faster computation
        dataset = numpy.array(dataset)
        # initialize random centroids
        centroids = self.__pick_cluster_centers(dataset)
        self.centroids = centroids
        # iterate through epochs
        for i in range(epochs_to_train):
            # assign cluster labels
            cluster_labels = self.__assign_cluster_labels(dataset, centroids)

            # update centroids
            centroids = self.__update_centroids(dataset, cluster_labels)

            # convergence prints
            dist = self.__euclidean_distance(self.centroids, centroids)
            print(f"Iteration {i}, distances: {dist}, mean distance {dist.mean()}")

            self.centroids = centroids

        # Compute final labels
        cluster_labels = self.__assign_cluster_labels(dataset, self.centroids)

        #Loop through each label and assign its corresponding data
        data: ClusteredDataset = {}
        for label in range(self.n_clusters):
            idx = np.where(cluster_labels == label)
            cluster_data = dataset[idx]
            cluster_data = list(map(tuple, cluster_data))
            data.setdefault(label, cluster_data)

        return data


def main():
    # Generate some data.
    gold_standard, embeddings = generate_data(clusters=N_CLUSTERS)

    # Initialize the model.
    clf = ClusteringModel(n_clusters=N_CLUSTERS)
    # Get the clusters.
    predictions = clf.fit(embeddings, epochs_to_train=10)

    # Evaluate the clustering.
    evaluate(gold_standard, predictions)


if __name__ == "__main__":
    main()
