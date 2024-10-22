import pytest
import numpy as np
import random_insertion as insertion
import math


@pytest.mark.parametrize("scale", [2, 20, 100])
@pytest.mark.parametrize("use_default_order", [True, False])
def test_tsp_insertion(use_default_order, scale):
    if use_default_order:
        order = None
    else:
        order = np.arange(0, scale, dtype=np.uint32)
        np.random.shuffle(order)
    cities = np.random.rand(scale, 2)
    route, cost = insertion.tsp_random_insertion(cities, order)
    assert (route < scale).all()
    assert len({x.item() for x in route}) == scale
    assert cost > 0


@pytest.mark.parametrize("instances", [1, 10])
@pytest.mark.parametrize("order_type", [None, 'shared', 'all'])
def test_tsp_insertion_parallel(order_type, instances, scale=100):
    if order_type is None:
        order = None
    elif order_type == 'shared':
        order = np.arange(0, scale, dtype=np.uint32)
        np.random.shuffle(order)
    else:
        order = []
        _order = np.arange(0, scale, dtype=np.uint32)
        for _ in range(instances):
            np.random.shuffle(_order)
            order.append(_order.copy())
        order = np.stack(order)

    cities = np.random.randn(instances, scale, 2)
    routes = insertion.tsp_random_insertion_parallel(cities, order)
    assert tuple(routes.shape) == (instances, scale)
    assert (routes < scale).all()
    assert len({x.item() for x in routes[-1]}) == scale


def test_tsp_insertion_line(scale=100):
    pos = np.arange(0, scale, dtype=np.float32)
    cities = np.stack([pos, np.zeros_like(pos)]).T
    route, cost = insertion.tsp_random_insertion(cities)
    assert len(route) == scale
    assert abs(cost - 2*(scale-1)) < 1e-5


def test_tsp_insertion_triangle(scale=100):
    pos = np.arange(0, scale, dtype=np.float32)
    pos = np.concatenate((pos, np.zeros_like(pos)))
    cities = np.stack([pos, pos[::-1]]).T
    route, cost = insertion.tsp_random_insertion(cities)
    assert len(route) == scale*2
    assert abs(cost - (2+math.sqrt(2))*(scale-1)) < 1e-5


def test_tsp_insertion_single_dot(scale=10):
    cities = np.zeros((scale, 2))
    cities.fill(math.pi)
    route, cost = insertion.tsp_random_insertion(cities)
    assert len(route) == scale
    assert cost == 0
