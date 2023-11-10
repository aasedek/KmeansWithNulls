import unittest
import numpy as np
from KmeansWithNulls.KmeansWithNulls import KmeansWithNulls 

class TestKMeans(unittest.TestCase):

    def setUp(self):
        # Setup a valid input dataset for tests
        self.X = np.array([[1, 2], [1, np.nan], [1, 0],
                           [10, 2], [10, 4], [10, 0]])

    def test_initialization(self):
        # Test if the KMeans class initializes correctly with valid parameters
        kmeans = KmeansWithNulls(n_clusters=2, max_iter=300, random_state=42)
        self.assertEqual(kmeans.n_clusters, 2)
        self.assertEqual(kmeans.max_iter, 300)
        self.assertEqual(kmeans.random_state, 42)

    def test_invalid_initialization(self):
        # Test if the KMeans class raises an error with invalid parameters
        with self.assertRaises(ValueError):
            KmeansWithNulls(n_clusters=-1, max_iter=300, random_state=42)
        with self.assertRaises(ValueError):
            KmeansWithNulls(n_clusters='invalid', max_iter=300, random_state=42)
        with self.assertRaises(ValueError):
            KmeansWithNulls(n_clusters=2, max_iter=-300, random_state=42)

    def test_check_input(self):
        # Test if the _check_input method works correctly
        kmeans = KmeansWithNulls(n_clusters=2)
        kmeans._check_input(self.X)  # Should pass without errors

        # Test various invalid inputs
        with self.assertRaises(ValueError):
            kmeans._check_input(np.array([]))
        with self.assertRaises(ValueError):
            kmeans._check_input(np.array([[np.nan, np.nan], [np.nan, np.nan]]))
        with self.assertRaises(ValueError):
            kmeans._check_input('invalid')
        with self.assertRaises(ValueError):
            kmeans._check_input(np.array([1, 2, 3]))

    def test_fit(self):
        # Test if the fit method works without raising errors
        kmeans = KmeansWithNulls(n_clusters=2)
        kmeans.fit(self.X)  # Should pass without errors

        # Check if centroids and labels are created
        self.assertIsNotNone(kmeans.centroids)
        self.assertEqual(len(kmeans.labels_), len(self.X))

    def test_predict(self):
        # Test if the predict method works correctly
        kmeans = KmeansWithNulls(n_clusters=2)
        kmeans.fit(self.X)
        predictions = kmeans.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))

    def test_empty_predict(self):
        # Test predict on empty input
        kmeans = KmeansWithNulls(n_clusters=2)
        kmeans.fit(self.X)
        with self.assertRaises(ValueError):
            kmeans.predict(np.array([]))


if __name__ == '__main__':
    unittest.main()
