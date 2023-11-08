# KmeansWithNulls (KWN) Clustering Implementation

This repository contains a Python implementation of the KMeans clustering algorithm which includes support for handling missing values in the dataset. The implementation allows for specifying the number of clusters, maximum iterations, and the random state for reproducibility.

## Features

- KMeans clustering.
- Handles missing data (NaN values) in the dataset.
- Customizable number of clusters and iterations.
- Utilizes NumPy for efficient numerical computations.

## Installation

No installation is required, just clone this repository using the following command:

```bash
git clone https://github.com/aasedek/KmeansWithNulls.git
```

## Usage

To use the KmeansWithNulls class, import it into your Python script and create an instance of the class. Then call the `fit` method with your dataset:

```python
from KmeansWithNulls import KmeansWithNulls
import numpy as np

# Example dataset
data = np.array([[1, 2], [1, 4], [1, 0],
                 [10, 2], [10, 4], [10, 0]])

# Initialize the KmeansWithNulls class
kmeans_with_nulls = KmeansWithNulls(n_clusters=2, max_iter=300, random_state=42)

# Fit the model to your data
kmeans_with_nulls.fit(data)

# Predict the clusters
labels = kmeans.predict(data)
```

## Example:

```python
import numpy as np
import pandas as pd

# Create a synthetic dataset with 100 points and 2 features
np.random.seed(42)
X_demo = np.random.rand(100, 2) * 10  # Scale the features by 10 for better visualization

# Introduce NaN values randomly in the dataset
nan_indices = np.random.choice(np.arange(X_demo.size), replace=False, size=10)
X_demo.ravel()[nan_indices] = np.nan

# Convert the array to a DataFrame for a better view of the NaN values
X_demo_df = pd.DataFrame(X_demo, columns=['Feature1', 'Feature2'])

# Instantiate the KMeansWithNulls class
kmeans_with_nulls = KmeansWithNulls(n_clusters=3)

# Fit the model to the data with NaNs
kmeans_with_nulls.fit(X_demo)

# Now let's predict the cluster labels for the dataset
predictions_with_nulls = kmeans_with_nulls.predict(X_demo)

# Output the centroids and predictions
centroids_with_nulls = kmeans_with_nulls.centroids
predictions_with_nulls

# Display the centroids and the first few predictions
print("Centroids:\n", centroids_with_nulls)
print("\nFirst 10 Predictions:\n", predictions_with_nulls[:10])

# Print the first few rows of the DataFrame with NaN values
X_demo_df.head()

# Plotting results

import matplotlib.pyplot as plt

# Plotting the clusters with nulls
plt.figure(figsize=(8, 6))

# Plot each cluster with available data, ignoring NaNs
for i in range(kmeans_with_nulls.n_clusters):
    # Select data points without NaNs that are assigned to the current cluster
    cluster_data = X_demo[~np.isnan(X_demo).any(axis=1) & (predictions_with_nulls == i)]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i+1}', s=50)

# Plot the centroids, excluding NaNs for plotting purposes
for idx, centroid in enumerate(centroids_with_nulls):
    if not np.isnan(centroid).any():
        plt.scatter(centroid[0], centroid[1], c='black', s=200, marker='x', label=f'Centroid {idx+1}')

plt.title('K-Means Clustering with Nulls')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

```
![Kmeans_with_nulls](resources/kmeans_with_nulls_clustering.JPG)


## Contributing
Contributions to improve this implementation are welcome. Before creating a pull request, please ensure your code follows the existing code structure and style.

## License
This project is open-sourced under the MIT License. See the LICENSE file for details.

## Contact
For questions and feedback, please reach out to me at arthur.sedek@gmail.com

## Acknowledgements
This implementation was inspired by the KMeans algorithm from scikit-learn.
