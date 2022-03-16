import numpy as np
from sklearn import datasets


def generate_data(cluster_std: int, n_data: int = 10) -> np.ndarray:
    return datasets.make_blobs(
        n_samples=n_data * 2,
        n_features=2,
        centers=2,
        cluster_std=cluster_std,
        center_box=(-20, 20),
    )
