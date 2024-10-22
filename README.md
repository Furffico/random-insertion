# Random-Insertion

A Python library for performing fast random insertion on TSP (Travelling Salesman Problem) and SHPP (Shortest Hamiltonian Path Problem) instances, originally a part of the [GLOP](https://github.com/henry-yeh/GLOP/tree/e2927170a8e6fa73563d1741690825dfae4f65f2/utils/insertion) codebase.

## Installation
### Dependencies
- python >= 3.7
- numpy >= 1.21

### Via PyPI
```bash
$ pip install random-insertion
```

### Build from source
```bash
$ git clone https://github.com/Furffico/random-insertion.git
$ cd random-insertion
$ pip install .
```

## Usages

For performing random insertion on multiple TSP instances in parallel:
```python
import numpy as np
import random_insertion as insertion

problem_scale = 50
num_instances = 10
coordinates = np.random.randn(num_instances, problem_scale, 2)
routes = insertion.tsp_random_insertion_parallel(coordinates, threads=4)
for route in routes:
    print(*route)
```

Despite the name, the program itself is deterministic in nature. Given the same instances and insertion order, the program will output identical routes. If you would like to add stochasticity to the outputs, please provide shuffled insertion orders like this:
```python
...
coordinates = np.random.randn(1, problem_scale, 2).repeat(num_instances, 0)
orders = np.arange(problem_scale, dtype=np.uint32).reshape(1, -1).repeat(num_instances, 0)
for i in range(num_instances):
    np.random.shuffle(orders[i])

routes = insertion.tsp_random_insertion_parallel(coordinates, orders, threads=4)
```

### Available methods

```python
# Recommended (threads=0 to automatically determine suitable values):
routes = tsp_random_insertion_parallel(coords, orders, threads=0)
routes = shpp_random_insertion_parallel(coords, orders, threads=0)
routes = atsp_random_insertion_parallel(distances, orders, threads=0)
routes = ashpp_random_insertion_parallel(distances, orders, threads=0)

# For backward compatibility with GLOP:
route, cost = tsp_random_insertion(coords, order)
route, cost = atsp_random_insertion(distances, order)

# Not tested:
route = cvrp_random_insertion(coords, depot_pos, demands, capacity, order, exploration = 1.0)
route = cvrplib_random_insertion(coords, demands, capacity, order, exploration = 1.0)
```
