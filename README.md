# KmeansWithNulls Clustering Implementation

This repository contains a Python implementation of the KMeans clustering algorithm which includes support for handling missing values in the dataset. The implementation allows for specifying the number of clusters, maximum iterations, and the random state for reproducibility.

## Features

- KMeans clustering.
- Handles missing data (NaN values) in the dataset.
- Customizable number of clusters and iterations.
- Utilizes NumPy for efficient numerical computations.

## Installation

No installation is required, just clone this repository using the following command:

(```bash
git clone https://github.com/aasedek/KmeansWithNulls.git
```)

## Usage

To use the KmeansWithNulls class, import it into your Python script and create an instance of the class. Then call the `fit` method with your dataset:

(```python
from KmeansWithNulls import KmeansWithNulls
import numpy as np

# Example dataset
data = np.array([[1, 2], [1, 4], [1, 0],
                 [10, 2], [10, 4], [10, 0]])

# Initialize the KMeans class
kmeans = KMeans(n_clusters=2, max_iter=300, random_state=42)

# Fit the model to your data
kmeans.fit(data)

# Predict the clusters
labels = kmeans.predict(data)
```)

## Contributing
Contributions to improve this implementation are welcome. Before creating a pull request, please ensure your code follows the existing code structure and style.

## License
This project is open-sourced under the MIT License. See the LICENSE file for details.

## Contact
For questions and feedback, please reach out to me at arthur.sedek@gmail.com

## Acknowledgements
This implementation was inspired by the KMeans algorithm from scikit-learn.
