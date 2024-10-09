from typing import Tuple

import numpy as np
import numpy.typing as npt

UInt32Array = npt.NDArray[np.uint32]
Float32Array = npt.NDArray[np.float32]

def random(
    instance: Float32Array,
    order: UInt32Array,
    is_euclidean: bool,
    out: UInt32Array,
) -> float: ...

def random_parallel(
    instance: Float32Array,
    order: UInt32Array,
    is_euclidean: bool,
    num_threads: int,
    out: UInt32Array,
) -> None: ...

def cvrp_random(
    customerpos: Float32Array,
    depotx: float,
    depoty: float,
    demands: UInt32Array,
    capacity: int,
    order: UInt32Array,
    exploration: float,
) -> Tuple[UInt32Array, UInt32Array]: ...
