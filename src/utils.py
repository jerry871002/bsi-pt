import pickle
from pathlib import Path
from typing import List, Union

import numpy as np


def normalize_distribution(
    distribution: Union[np.ndarray, List[float]],
    min_value: float
) -> np.ndarray:
    distribution = np.array(distribution).astype(float)
    distribution[distribution < min_value] = min_value
    return distribution / np.sum(distribution)

def store_as_pickle(obj: object, path: Union[Path, str]) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(pickle_file: Union[Path, str]) -> object:
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    return data
