import pytest
import numpy as np
import random_insertion as insertion

def validate_cvrp_routes(scale, routes, demands, capacity):
    cities = set(range(scale))
    for route in routes:
        total_demand = 0
        for city in route:
            assert city in cities, f"City {city} visited more than once"
            cities.remove(city)
            total_demand += demands[city]
        assert total_demand <= capacity, "Route total demand exceeds capacity"
    assert len(cities) == 0, "Some cities were not visited"

@pytest.mark.parametrize("scale", [20, 100])
@pytest.mark.parametrize("use_default_order", [True, False])
def test_cvrp_insertion(scale, use_default_order):
    if use_default_order:
        order = None
    else:
        order = np.arange(0, scale, dtype=np.uint32)
        np.random.shuffle(order)

    pos = np.random.randn(scale, 2)
    depotpos = np.random.randn(2)
    demands = np.random.randint(1, 6, size = scale)
    capacity = 20
    result = insertion.cvrp_random_insertion(pos, depotpos, demands, capacity, order)
    validate_cvrp_routes(scale, result, demands, capacity)


@pytest.mark.parametrize("scale", [100])
@pytest.mark.parametrize("instances", [1, 10])
@pytest.mark.parametrize("order_type", [None, 'shared', 'all'])
@pytest.mark.parametrize("shared_capacity", [True, False])
def test_cvrp_insertion_parallel(scale, instances, order_type, shared_capacity):
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

    pos = np.random.randn(instances, scale, 2)
    depotpos = np.random.randn(instances, 2)
    demands = np.random.randint(1, 6, size = (instances, scale))
    if shared_capacity:
        capacity = 20
    else:
        capacity = np.random.randint(20, 40, size = instances)
    result = insertion.cvrp_random_insertion_parallel(pos, depotpos, demands, capacity, order)
    for i in range(instances):
        validate_cvrp_routes(scale, result[i], demands[i], capacity if isinstance(capacity, int) else capacity[i])