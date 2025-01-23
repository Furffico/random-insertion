# random-insertion
<a href="https://pypi.org/project/random-insertion"><img alt="PyPI" src="https://img.shields.io/pypi/v/random-insertion?logo=pypi"></a>
<a href="https://pypi.org/project/random-insertion"><img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/random-insertion"></a>
<a href="https://github.com/Furffico/random-insertion"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Furffico/random-insertion"></a>

`random-insertion` is a Python library for performing fast random insertion on TSP (Travelling Salesman Problem) and SHPP (Shortest Hamiltonian Path Problem) instances, originally a part of the [GLOP](https://github.com/henry-yeh/GLOP/tree/e2927170a8e6fa73563d1741690825dfae4f65f2/utils/insertion)* codebase.

\* Ye, H., Wang, J., Liang, H., Cao, Z., Li, Y., & Li, F. (2024). GLOP: Learning Global Partition and Local Construction for Solving Large-Scale Routing Problems in Real-Time. *Proceedings of the AAAI Conference on Artificial Intelligence, 38*(18), 20284-20292. https://doi.org/10.1609/aaai.v38i18.30009

## Installation
### Supported environments
- python >= 3.7
- numpy >= 1.21
- Linux and Windows

### Install from PyPI
```bash
$ pip install random-insertion
```

### Build from source
```bash
$ git clone https://github.com/Furffico/random-insertion.git
$ cd random-insertion
$ pip install . # add `-e` for development mode
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

Despite the name, the program itself is deterministic in nature.
Given the same instances and insertion orders, the program will output identical routes.
If you would like to add stochasticity to the outputs, please provide shuffled insertion orders like this:
```python
...
coordinates = np.random.randn(1, problem_scale, 2).repeat(num_instances, 0)
orders = np.arange(problem_scale, dtype=np.uint32).reshape(1, -1).repeat(num_instances, 0)
for i in range(num_instances):
    np.random.shuffle(orders[i])
routes = insertion.tsp_random_insertion_parallel(coordinates, orders)
```

### Available methods

```python
# Recommended (threads=0 to automatically determine suitable values):
routes = tsp_random_insertion_parallel(coords, orders, threads=0)
routes = shpp_random_insertion_parallel(coords, orders, threads=0)
routes = atsp_random_insertion_parallel(distances, orders, threads=0)
routes = ashpp_random_insertion_parallel(distances, orders, threads=0)
routes = cvrp_random_insertion_parallel(coords, depot_coords, demands, capacity, order, threads=0)

# For backward compatibility with GLOP:
route, cost = tsp_random_insertion(coords, order)
route, cost = atsp_random_insertion(distances, order)
route = cvrp_random_insertion(coords, depot_pos, demands, capacity, order)
route = cvrplib_random_insertion(coords, demands, capacity, order)
```

### Running tests
```bash
$ git clone https://github.com/Furffico/random-insertion.git
$ cd random-insertion
$ pip install -e . pytest
$ pytest                  # run tests
```
