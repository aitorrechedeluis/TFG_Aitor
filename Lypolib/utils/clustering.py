import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter
from skimage.morphology import remove_small_objects
import warnings

from ..config import default

class KMeansClustering:

    def __init__(self, kmatrix):

        self.matrix = np.nanmean(kmatrix, axis=2)
        self.max_x, self.max_y = self.matrix.shape

    def KMeansCluster(self):

        model = KMeans(n_clusters=default.N_CLUSTERS, n_init=10, random_state=0)

        data = self.matrix.reshape(-1)

        # Masking nan values
        mask = ~np.isnan(data)
        labels = np.full(data.shape[0], -1, dtype=int)

        # Fitting clusters
        valid_data = data[mask].reshape(-1, 1)
        labels[mask] = model.fit_predict(valid_data)
        cluster_map = labels.reshape(self.max_x, self.max_y)

        # Spatial clean-up
        valid_mask = cluster_map != -1

        filtered = median_filter(cluster_map, size=3)
        cluster_map[valid_mask] = filtered[valid_mask]

        # Removing small objects
        counts = np.bincount(labels[mask])
        largest = np.argmax(counts)
        tissue_mask = (cluster_map == largest)
        tissue_mask = remove_small_objects(tissue_mask, min_size=128)

        cluster_map[~tissue_mask] = -1

        return cluster_map, largest