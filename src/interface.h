#ifndef __RANDOM_INSERTION_MODULE_HEAD

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

PyObject* tsp_insertion_random(PyObject *self, PyObject *args);
PyObject* tsp_insertion_random_parallel(PyObject *self, PyObject *args);
PyObject* cvrp_insertion_random(PyObject *self, PyObject *args);

#include "interface_tsp.h"
#include "interface_cvrp.h"

#define __RANDOM_INSERTION_MODULE_HEAD
#endif