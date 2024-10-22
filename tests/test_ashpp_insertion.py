import pytest
import numpy as np
import random_insertion as insertion


@pytest.mark.parametrize("instances,scale", [(1, 100), (1, 1000), (10, 100)])
@pytest.mark.parametrize("order_type", [None, 'shared', 'all'])
def test_ashpp_insertion_parallel(order_type, instances, scale):
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
    routes = insertion.ashpp_random_insertion_parallel(distances, order)
    assert tuple(routes.shape) == (instances, scale)
    assert (routes < scale).all()
    assert len({x.item() for x in routes[-1]}) == scale
