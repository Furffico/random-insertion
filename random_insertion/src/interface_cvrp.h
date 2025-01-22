#ifndef __RANDOM_INSERTION_INTERFACE_CVRP
#define __RANDOM_INSERTION_INTERFACE_CVRP

#include "head_cvrp.h"
#include <Python.h>
#include "numpy/arrayobject.h"
#include <iostream>

static PyObject*
cvrp_insertion_random(PyObject *self, PyObject *args)
{
    /* ----------------- read cities' position from PyObject ----------------- */
    PyObject *pycities, *pyorder, *pydemands, *pyoutorder, *pyoutsep;
    float depotx, depoty;
    unsigned capacity;
    // positions depotx depoty demands capacity order
    if (!PyArg_ParseTuple(args, "OffOIOOO", &pycities, &depotx, &depoty, &pydemands, &capacity, &pyorder, &pyoutorder, &pyoutsep))
        return NULL;
    if (!PyArray_Check(pycities) || !PyArray_Check(pyorder) || !PyArray_Check(pydemands) ||!PyArray_Check(pyoutorder) || !PyArray_Check(pyoutsep))
        return NULL;
    
    PyArrayObject *pyarrcities = (PyArrayObject *)pycities, *pyarrorder = (PyArrayObject *)pyorder;
    PyArrayObject *pyarrdemands = (PyArrayObject *)pydemands, *pyarroutorder = (PyArrayObject *)pyoutorder, *pyarroutsep = (PyArrayObject *)pyoutsep;

    #ifndef SKIPCHECK
    if (PyArray_NDIM(pyarrcities) != 2 || PyArray_TYPE(pyarrcities) != NPY_FLOAT32
        || PyArray_NDIM(pyarrorder) != 1 || PyArray_TYPE(pyarrorder) != NPY_UINT32
        || PyArray_NDIM(pyarrdemands) != 1 || PyArray_TYPE(pyarrdemands) != NPY_UINT32)
        return NULL;
    #endif

    npy_intp *shape = PyArray_SHAPE(pyarrcities);
    unsigned citycount = (unsigned)shape[0];
    unsigned maxroutecount = (unsigned)PyArray_SHAPE(pyarroutsep)[0];
    float *cities = (float *)PyArray_DATA(pyarrcities);
    unsigned *order = (unsigned *)PyArray_DATA(pyarrorder);
    unsigned *demands = (unsigned *)PyArray_DATA(pyarrdemands);
    unsigned *outorder = (unsigned *)PyArray_DATA(pyarroutorder);
    unsigned *outsep = (unsigned *)PyArray_DATA(pyarroutsep);
    float depotpos[2] = {depotx, depoty};

    /* ---------------------------- random insertion ---------------------------- */
    CVRPInstance cvrpi = CVRPInstance(citycount, cities, demands, depotpos, capacity, outorder, outsep, maxroutecount);
    CVRPInsertion ins = CVRPInsertion(&cvrpi);

    ins.randomInsertion(order);
    return Py_None;
}
#endif