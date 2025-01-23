from typing import Tuple, overload

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

@overload
def cvrp_random_parallel(
    customerpos: Float32Array,
    depotpos: Float32Array,
    demands: UInt32Array,
    capacity: UInt32Array,
    order: UInt32Array,
    num_threads: int,
    outorder: UInt32Array,
    outsep: UInt32Array,
) -> None: ...

@overload
def cvrp_random_parallel(
    customerpos: Float32Array,
    depotpos: Float32Array,
    demands: UInt32Array,
    capacity: int,
    order: UInt32Array,
    num_threads: int,
    outorder: UInt32Array,
    outsep: UInt32Array,
) -> None: ...

def shpp_random_parallel(
    instance: Float32Array,
    order: UInt32Array,
    is_euclidean: bool,
    num_threads: int,
    out: UInt32Array,
) -> None: ...