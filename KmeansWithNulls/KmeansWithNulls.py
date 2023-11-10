import numpy as np

class KmeansWithNulls:
    def __init__(self, n_clusters=2, max_iter=300, random_state=42):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None

    def _check_input(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        if np.isnan(X).all():
            raise ValueError("Input data must not be all NaNs")
        if X.size == 0:
            raise ValueError("Input data must not be empty")
        if len(X.shape) != 2:
            raise ValueError("Input data must be 2-dimensional")

    def _init_centroids(self, X):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def _closest_centroid(self, X):
        distances = np.zeros((self.n_clusters, X.shape[0]))
        for i, centroid in enumerate(self.centroids):
            # Repeat the centroid to match the shape of X, ignoring NaN in both
            non_nan_mask = ~np.isnan(X)
            repeated_centroid = np.repeat(centroid.reshape(1, -1), X.shape[0], axis=0)
            valid_centroid_mask = ~np.isnan(repeated_centroid)

            # Combined mask of non-NaN entries for both data points and centroid
            combined_mask = non_nan_mask & valid_centroid_mask
            
            # Perform subtraction only over the valid entries
            diff = np.zeros_like(X)
            diff[combined_mask] = X[combined_mask] - repeated_centroid[combined_mask]
            
            # Square the differences and sum over features, ignoring NaNs
            squared_diff = diff**2
            distances[i, :] = np.nansum(squared_diff, axis=1)

        # Return the index of the closest centroid for each data point
        return np.argmin(distances, axis=0)

    def _compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_data = X[labels == i]
            # Calculate the mean for each feature in the cluster, ignoring NaNs
            for j in range(X.shape[1]):
                feature_data = cluster_data[:, j]
                # If all values for a feature are NaN, set the centroid feature to NaN
                if np.isnan(feature_data).all():
                    centroids[i, j] = np.nan
                else:
                    centroids[i, j] = np.nanmean(feature_data)
        return centroids


    def _converged(self, centroids_old, centroids):
        # We also need to modify the convergence check to ignore NaNs
        return np.all(np.nan_to_num(centroids_old) == np.nan_to_num(centroids))

    def fit(self, X):
        self._check_input(X)  # Check for valid input
        # Replace missing values with np.nan for consistency
        X = np.where(np.isnan(X), np.nan, X)
        self.centroids = self._init_centroids(X)
        
        for _ in range(self.max_iter):
            centroids_old = self.centroids.copy()

            labels = self._closest_centroid(X)
            self.centroids = self._compute_centroids(X, labels)

            if self._converged(centroids_old, self.centroids):
                break

        self.labels_ = labels
        return self

    def predict(self, X):
        self._check_input(X)  # Check for valid input
        # Predict should also handle NaNs
        X = np.where(np.isnan(X), np.nan, X)
        return self._closest_centroid(X)
