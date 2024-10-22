#ifndef __RANDOM_INSERTION_INTERFACE_SHPP
#define __RANDOM_INSERTION_INTERFACE_SHPP

#include "interface_tsp.h"

static PyObject*
shpp_insertion_random_parallel(PyObject *self, PyObject *args)
{
    PyObject *pycities, *pyorder, *pyout;
    int isEuclidean = 1, numThreads_ = 0;
    if (!PyArg_ParseTuple(args, "OOpiO", &pycities, &pyorder, &isEuclidean, &numThreads_, &pyout))
        return NULL;

    TaskList<SHPPInsertion> solvers = read_tsp_instance<SHPPInsertion>(pycities, pyorder, isEuclidean, pyout, true);
    solvers.solve_parallel(numThreads_);
    return Py_None;
}
#endif