# %%
import numpy as np
from typing import Any


# TODO: implement the PCA with numpy
# Note that you are not allowed to use any existing PCA implementation from sklearn or other libraries.
class PrincipalComponentAnalysis:
    def __init__(self, n_components: int) -> None:
        """_summary_

        Parameters
        ----------
        n_components : int
            The number of principal components to be computed. This value should be less than or equal to the number of features in the dataset.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    # TODO: implement the fit method
    def fit(self, X: np.ndarray):
        """
        Fit the model with X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
                # Centering the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Computing covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sorting eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:,idx]
        
        # Selecting the top n_components
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)


# TODO: implement the LDA with numpy
# Note that you are not allowed to use any existing LDA implementation from sklearn or other libraries.
class LinearDiscriminantAnalysis:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.

        Hint:
        -----
        To implement LDA with numpy, follow these steps:
        1. Compute the mean vectors for each class.
        2. Compute the within-class scatter matrix.
        3. Compute the between-class scatter matrix.
        4. Compute the eigenvectors and corresponding eigenvalues for the scatter matrices.
        5. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues to form a d×k dimensional matrix W.
        6. Use this d×k eigenvector matrix to transform the samples onto the new subspace.
        """
        n_features = X.shape[1]
        class_labels = np.unique(y)
        
        # Compute overall mean
        mean_overall = np.mean(X, axis=0)
        
        # Within-class scatter matrix SW and between-class scatter matrix SB
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        
        for label in class_labels:
            X_class = X[y == label]
            mean_class = np.mean(X_class, axis=0)
            
            # Within-class scatter matrix for current class
            SW += np.dot((X_class - mean_class).T, (X_class - mean_class))
            
            # Between-class scatter
            n_class = X_class.shape[0]
            mean_diff = (mean_class - mean_overall).reshape(n_features, 1)
            SB += n_class * np.dot(mean_diff, mean_diff.T)
        
        # Solve the generalized eigenvalue problem for SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(SW).dot(SB))
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Select the top n_components
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        return np.dot(X, self.components)


# TODO: Generating adversarial examples for PCA.
# We will generate adversarial examples for PCA. The adversarial examples are generated by creating two well-separated clusters in a 2D space. Then, we will apply PCA to the data and check if the clusters are still well-separated in the transformed space.
# Your task is to generate adversarial examples for PCA, in which
# the clusters are well-separated in the original space, but not in the PCA space. The separabilit of the clusters will be measured by the K-means clustering algorithm in the test script.
#
# Hint:
# - You can place the two clusters wherever you want in a 2D space.
# - For example, you can use `np.random.multivariate_normal` to generate the samples in a cluster. Repeat this process for both clusters and concatenate the samples to create a single dataset.
# - You can set any covariance matrix, mean, and number of samples for the clusters.
class AdversarialExamples:
    def __init__(self) -> None:
        pass

    def pca_adversarial_data(self, n_samples, n_features):
        """Generate adversarial examples for PCA

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        n_features : int
            The number of features.

        Returns
        -------
        X: ndarray of shape (n_samples, n_features)
            Transformed values.

        y: ndarray of shape (n_samples,)
            Cluster IDs. y[i] is the cluster ID of the i-th sample.

        """
        n_samples_per_cluster = n_samples // 2
        mean1 = np.zeros(n_features)
        mean1[0] = 1  
        cov1 = np.eye(n_features)  
        
        mean2 = np.zeros(n_features)
        mean2[0] = 4  
        cov2 = np.eye(n_features) * 0.5  
        
        # Generate samples for each cluster
        cluster1 = np.random.multivariate_normal(mean1, cov1, n_samples_per_cluster)
        cluster2 = np.random.multivariate_normal(mean2, cov2, n_samples_per_cluster)
        
        # Concatenate clusters to form the dataset
        X = np.vstack((cluster1, cluster2))
        y = np.array([0]*n_samples_per_cluster + [1]*n_samples_per_cluster)
        return X, y