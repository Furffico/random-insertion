try:
    from . import _core
except:
    import _core
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Optional, List

UInt32Array = npt.NDArray[np.uint32]
Float32Array = npt.NDArray[np.float32]
IntegerArray = npt.NDArray[Union[np.int_, np.uint]]
FloatPointArray = npt.NDArray[Union[np.float16, np.float32, np.float64]]


def _tsp_get_parameters(
    cities: FloatPointArray,
    order: Optional[IntegerArray] = None,
    batched=False,
    euclidean=True
) -> Tuple[Tuple[Float32Array, UInt32Array, bool], UInt32Array]:
    if batched:
        assert len(cities.shape) == 3
        batch_size, citycount, n = cities.shape
        out = np.zeros((batch_size, citycount), dtype=np.uint32)
    else:
        assert len(cities.shape) == 2
        citycount, n = cities.shape
        out = np.zeros(citycount, dtype=np.uint32)

    assert (n == 2) if euclidean else (n == citycount)

    if order is None:
        order = np.arange(citycount, dtype=np.uint32)
    else:
        if batched and len(order.shape) == 2:
            assert tuple(order.shape) == (batch_size, citycount)
        else:
            assert len(order.shape) == 1 and order.shape[0] == citycount

    _order = np.ascontiguousarray(order, dtype=np.uint32)
    _cities = np.ascontiguousarray(cities, dtype=np.float32)
    return (_cities, _order, euclidean), out


def tsp_random_insertion(
    cities: FloatPointArray,
    order: Optional[IntegerArray] = None
) -> Tuple[UInt32Array, float]:
    args, out = _tsp_get_parameters(cities, order)
    cost = _core.random(*args, out)
    assert cost >= 0
    return out, cost


def tsp_random_insertion_parallel(
    cities: FloatPointArray,
    order: Optional[IntegerArray] = None,
    threads: int = 0
) -> UInt32Array:
    args, out = _tsp_get_parameters(cities, order, batched=True)
    _core.random_parallel(*args, threads, out)
    return out


def atsp_random_insertion(
    distmap: FloatPointArray,
    order: Optional[IntegerArray] = None
) -> Tuple[UInt32Array, float]:
    args, out = _tsp_get_parameters(distmap, order, euclidean=False)
    cost = _core.random(*args, out)
    assert cost >= 0
    return out, cost


def atsp_random_insertion_parallel(
    distmap: FloatPointArray,
    order: Optional[IntegerArray] = None,
    threads: int = 0
) -> UInt32Array:
    args, out = _tsp_get_parameters(
        distmap, order, batched=True, euclidean=False)
    _core.random_parallel(*args, threads, out)
    return out


def shpp_random_insertion_parallel(
    cities: FloatPointArray,
    order: Optional[IntegerArray] = None,
    threads: int = 0
) -> UInt32Array:
    args, out = _tsp_get_parameters(cities, order, batched=True)
    _core.shpp_random_parallel(*args, threads, out)
    return out

def ashpp_random_insertion_parallel(
    cities: FloatPointArray,
    order: Optional[IntegerArray] = None,
    threads: int = 0
) -> UInt32Array:
    args, out = _tsp_get_parameters(cities, order, batched=True, euclidean=False)
    _core.shpp_random_parallel(*args, threads, out)
    return out

def cvrp_random_insertion(
    customerpos: FloatPointArray,
    depotpos: FloatPointArray,
    demands: IntegerArray,
    capacity: int,
    order: Optional[IntegerArray] = None,
    exploration: float = 1.0
) -> List[UInt32Array]:
    assert len(customerpos.shape) == 2 and customerpos.shape[1] == 2
    assert isinstance(capacity, int)

    if isinstance(depotpos, tuple):
        assert len(depotpos) == 2
        depotx, depoty = depotpos
    else:
        assert len(depotpos.shape) == 1 and depotpos.shape[0] == 2
        depotx, depoty = depotpos[0].item(), depotpos[1].item()
    depotx, depoty = float(depotx), float(depoty)

    ccount = customerpos.shape[0]
    if order is None:
        # generate order
        dx, dy = (customerpos - np.array([[depotx, depoty]])).T
        phi = np.arctan2(dy, dx)
        order = np.argsort(phi).astype(np.uint32)
    else:
        assert len(order.shape) == 1 and order.shape[0] == ccount

    _order = np.ascontiguousarray(order, dtype=np.uint32)
    _customerpos = np.ascontiguousarray(customerpos, dtype=np.float32)
    _demands = np.ascontiguousarray(demands, dtype=np.uint32)

    outorder, sep = _core.cvrp_random(
        _customerpos, depotx, depoty, _demands, capacity, _order, exploration)
    routes = [outorder[i:j] for i, j in zip(sep, sep[1:])]
    return routes


def cvrplib_random_insertion(
    positions: FloatPointArray,
    demands: IntegerArray,
    capacity: int,
    order: Optional[IntegerArray] = None,
    exploration=1.0
) -> List[UInt32Array]:
    customerpos = positions[1:]
    depotpos = positions[0]
    demands = demands[1:]
    if order is not None:
        order = np.delete(order, order == 0) - 1
    routes = cvrp_random_insertion(
        customerpos, depotpos, demands, capacity, order, exploration)
    for r in routes:
        r += 1
    return routes
