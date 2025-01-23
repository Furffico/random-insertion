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

@pytest.mark.parametrize("scale", [20, 100, 1000])
@pytest.mark.parametrize("use_default_order", [True, False])
def test_cvrp_insertion(scale, use_default_order):
    if use_default_order:
        order = None
    else:
        order = np.arange(0, scale, dtype=np.uint32)
        np.random.shuffle(order)

    pos = np.random.randn(scale, 2)
    depotpos = np.random.randn(2)
    demands = np.random.randint(1, 7, size = scale)
    capacity = 30
    result = insertion.cvrp_random_insertion(pos, depotpos, demands, capacity, order)
    validate_cvrp_routes(scale, result, demands, capacity)