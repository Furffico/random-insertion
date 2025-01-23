#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

static PyObject* tsp_insertion_random(PyObject *self, PyObject *args);
static PyObject* tsp_insertion_random_parallel(PyObject *self, PyObject *args);
static PyObject* cvrp_insertion_random(PyObject *self, PyObject *args);
static PyObject* shpp_insertion_random_parallel(PyObject *self, PyObject *args);

// #define SKIPCHECK
#include "interface_tsp.h"
#include "interface_cvrp.h"
#include "interface_shpp.h"

static PyMethodDef InsertionMethods[] = {
    {"random", tsp_insertion_random, METH_VARARGS, "Execute random insertion on TSP."},
    {"random_parallel", tsp_insertion_random_parallel, METH_VARARGS, "Execute batched random insertion on TSP."},
    {"cvrp_random_parallel", cvrp_insertion_random_parallel, METH_VARARGS, "Execute batched random insertion on CVRP."},
    {"shpp_random_parallel", shpp_insertion_random_parallel, METH_VARARGS, "Execute batched random insertion on SHPP."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef insertionmodule = {
    PyModuleDef_HEAD_INIT,
    "insertion",
    NULL,
    -1,
    InsertionMethods};

PyMODINIT_FUNC
PyInit__core(void)
{
    PyObject *m = PyModule_Create(&insertionmodule);
    if (m == NULL) return NULL;
    import_array();

    return m;
}
