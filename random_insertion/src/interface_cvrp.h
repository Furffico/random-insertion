#ifndef __RANDOM_INSERTION_INTERFACE_CVRP
#define __RANDOM_INSERTION_INTERFACE_CVRP

#include "interface_common.h"

static PyObject*
cvrp_insertion_random(PyObject *self, PyObject *args)
{
    /* ----------------- read cities' position from PyObject ----------------- */
    PyObject *pycities, *pyorder, *pydemands;
    float depotx, depoty, exploration;
    unsigned capacity;
    // positions depotx depoty demands capacity order
    if (!PyArg_ParseTuple(args, "OffOIOf", &pycities, &depotx, &depoty, &pydemands, &capacity, &pyorder, &exploration))
        return NULL;
    if (!PyArray_Check(pycities) || !PyArray_Check(pyorder) || !PyArray_Check(pydemands))
        return NULL;
    
    PyArrayObject *pyarrcities = (PyArrayObject *)pycities, *pyarrorder = (PyArrayObject *)pyorder, *pyarrdemands = (PyArrayObject *)pydemands;

    #ifndef SKIPCHECK
    if (PyArray_NDIM(pyarrcities) != 2 || PyArray_TYPE(pyarrcities) != NPY_FLOAT32
        || PyArray_NDIM(pyarrorder) != 1 || PyArray_TYPE(pyarrorder) != NPY_UINT32
        || PyArray_NDIM(pyarrdemands) != 1 || PyArray_TYPE(pyarrdemands) != NPY_UINT32)
        return NULL;
    #endif

    npy_intp *shape = PyArray_SHAPE(pyarrcities);
    unsigned citycount = (unsigned)shape[0];
    float *cities = (float *)PyArray_DATA(pyarrcities);
    unsigned *order = (unsigned *)PyArray_DATA(pyarrorder);
    unsigned *demands = (unsigned *)PyArray_DATA(pyarrdemands);
    float depotpos[2] = {depotx, depoty};

    /* ---------------------------- random insertion ---------------------------- */
    CVRPInstance cvrpi = CVRPInstance(citycount, cities, demands, depotpos, capacity);
    CVRPInsertion ins = CVRPInsertion(&cvrpi);

    CVRPReturn *result = ins.randomInsertion(order, exploration);
    /* ----------------------- convert output to PyObject ----------------------- */
    npy_intp dims = citycount, dims2 = result->routes;
    PyObject *returntuple = PyTuple_Pack(2, 
        PyArray_SimpleNewFromData(1, &dims, NPY_UINT32, result->order),
        PyArray_SimpleNewFromData(1, &dims2, NPY_UINT32, result->routesep)
    );

    return returntuple;
}
#endif