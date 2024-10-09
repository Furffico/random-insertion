// #define SKIPCHECK
#include "interface.h"

static PyMethodDef InsertionMethods[] = {
    {"random", tsp_insertion_random, METH_VARARGS, "Execute random insertion on TSP."},
    {"random_parallel", tsp_insertion_random_parallel, METH_VARARGS, "Execute batched random insertion on TSP."},
    {"cvrp_random", cvrp_insertion_random, METH_VARARGS, "Execute random insertion on CVRP."},
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
    PyObject *m;
    m = PyModule_Create(&insertionmodule);
    if (m == NULL)
        return NULL;
    import_array();

    return m;
}
