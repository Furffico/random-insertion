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

def cvrp_random_insertion_parallel(
    customerpos: FloatPointArray,
    depotpos: FloatPointArray,
    demands: IntegerArray,
    capacity: Union[IntegerArray, int],
    order: Optional[IntegerArray] = None,
    threads: int = 0
) -> List[List[UInt32Array]]:
    assert len(customerpos.shape) == 3 and customerpos.shape[2] == 2
    batch_size, ccount, _ = customerpos.shape
    assert depotpos.shape == (batch_size, 2)
    assert demands.shape == (batch_size, ccount)
    assert (demands.max() <= capacity).all() and demands.min() > 0
    
    if order is None:
        # generate order
        dp = (customerpos - depotpos.reshape(batch_size, 1, 2))
        dx, dy = dp[..., 0], dp[..., 1]
        phi = np.arctan2(dy, dx)
        order = np.argsort(phi, axis=1).astype(np.uint32)
    else:
        assert order.shape == (batch_size, ccount) or order.shape == (ccount,)
    
    if not isinstance(capacity, int):
        assert capacity.shape == (batch_size,)
        _capacity = np.ascontiguousarray(capacity, dtype=np.uint32)
    else:
        _capacity = capacity

    _order = np.ascontiguousarray(order, dtype=np.uint32)
    _customerpos = np.ascontiguousarray(customerpos, dtype=np.float32)
    _depotpos = np.ascontiguousarray(depotpos, dtype=np.float32)
    _demands = np.ascontiguousarray(demands, dtype=np.uint32)
    max_routes = min((demands.sum(axis=-1) // capacity).max() + 10, ccount)

    outorder = np.ascontiguousarray(np.zeros((batch_size, ccount), dtype=np.uint32))
    outsep = np.ascontiguousarray(np.zeros((batch_size, max_routes), dtype=np.uint32))

    _core.cvrp_random_parallel(
        _customerpos,
        _depotpos,
        _demands,
        _capacity,
        _order,
        threads,
        outorder,
        outsep,
    )
    all_routes = []
    for idx in range(batch_size):
        routes = [outorder[idx, i:j] for i, j in zip(outsep[idx, :], outsep[idx, 1:]) if j>0]
        all_routes.append(routes)

    return all_routes

def cvrp_random_insertion(
    customerpos: FloatPointArray,
    depotpos: FloatPointArray,
    demands: IntegerArray,
    capacity: int,
    order: Optional[IntegerArray] = None,
) -> List[UInt32Array]:
    return cvrp_random_insertion_parallel(
        customerpos.reshape(1, *customerpos.shape),
        depotpos.reshape(1, *depotpos.shape),
        demands.reshape(1, *demands.shape),
        capacity,
        order.reshape(1, *order.shape) if order is not None else None,
        threads=0,
    )[0]

def cvrplib_random_insertion(
    positions: FloatPointArray,
    demands: IntegerArray,
    capacity: int,
    order: Optional[IntegerArray] = None,
) -> List[UInt32Array]:
    customerpos = positions[1:]
    depotpos = positions[0]
    demands = demands[1:]
    if order is not None:
        order = np.delete(order, order == 0) - 1
    routes = cvrp_random_insertion(customerpos, depotpos, demands, capacity, order)
    for r in routes:
        r += 1
    return routes
