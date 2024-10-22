import pytest
import numpy as np
import random_insertion as insertion


@pytest.mark.parametrize("scale", [2, 20, 100])
@pytest.mark.parametrize("use_default_order", [True, False])
def test_atsp_insertion(use_default_order, scale):
    if use_default_order:
        order = None
    else:
        order = np.arange(0, scale, dtype=np.uint32)
        np.random.shuffle(order)
    distances = np.random.rand(scale, scale)
    route, cost = insertion.atsp_random_insertion(distances, order)
    assert (route < scale).all()
    assert len({x.item() for x in route}) == scale
    assert cost > 0


@pytest.mark.parametrize("instances", [1, 10])
@pytest.mark.parametrize("order_type", [None, 'shared', 'all'])
def test_atsp_insertion_parallel(order_type, instances, scale=100):
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

    distances = np.random.rand(instances, scale, scale)
    routes = insertion.atsp_random_insertion_parallel(distances, order)
    assert tuple(routes.shape) == (instances, scale)
    assert (routes < scale).all()
    assert len({x.item() for x in routes[-1]}) == scale
