#ifndef __RANDOM_INSERTION_INTERFACE_TSP
#define __RANDOM_INSERTION_INTERFACE_TSP

#include "head_tsp_shpp.h"
#include "interface_common.h"

float get_tsp_insertion_result(TSPinstance *tspi){
    TSPInsertion ins = TSPInsertion(tspi);
    return ins.solve();
}

template<class Insertion>
TaskList<Insertion> read_tsp_instance(PyObject *pycities, PyObject *pyorder, bool isEuclidean, PyObject *pyout, bool batched){
    TaskList<Insertion> instances({});

    if (!PyArray_Check(pycities) || !PyArray_Check(pyorder)|| !PyArray_Check(pyout))
        return instances;
    
    PyArrayObject *pyarrcities = (PyArrayObject *)pycities, *pyarrorder = (PyArrayObject *)pyorder, *pyarrout = (PyArrayObject *)pyout;

    #ifndef SKIPCHECK
    if (PyArray_NDIM(pyarrcities) != 2+batched || PyArray_TYPE(pyarrcities) != NPY_FLOAT32
        || !(PyArray_NDIM(pyarrorder) == 1 || (batched && PyArray_NDIM(pyarrorder) == 2)) || PyArray_TYPE(pyarrorder) != NPY_UINT32
        || PyArray_NDIM(pyarrout) != 1+batched  || PyArray_TYPE(pyarrout) != NPY_UINT32)
        return instances;
    #endif

    npy_intp *shape = PyArray_SHAPE(pyarrcities);
    unsigned citycount, batchsize=1;
    bool shared_order=false;

    if(batched){
        shared_order = PyArray_NDIM(pyarrorder) == 1;
        batchsize = (unsigned)shape[0], citycount = (unsigned)shape[1];
        citycount = (unsigned)shape[1];
        #ifndef SKIPCHECK
        if ((unsigned)shape[2]!=(isEuclidean?2:citycount)
            || (shared_order && (unsigned)PyArray_SHAPE(pyarrorder)[0] != citycount) 
            || (!shared_order && ((unsigned)PyArray_SHAPE(pyarrorder)[0] != batchsize || (unsigned)PyArray_SHAPE(pyarrorder)[1] != citycount))
            || (unsigned)PyArray_SHAPE(pyarrout)[0] != batchsize || (unsigned)PyArray_SHAPE(pyarrout)[1] != citycount)
            return instances;
        #endif
        instances.reserve(batchsize);
    }else{
        citycount = (unsigned)shape[0];
        #ifndef SKIPCHECK
        if ((unsigned)shape[1]!=(isEuclidean?2:citycount)
            || (unsigned)PyArray_SHAPE(pyarrorder)[0]!=citycount 
            || (unsigned)PyArray_SHAPE(pyarrout)[0]!=citycount)
            return instances;
        #endif
    }

    float *cities = (float *)PyArray_DATA(pyarrcities);
    unsigned* order = (unsigned *)PyArray_DATA(pyarrorder);
    unsigned* out = (unsigned *)PyArray_DATA(pyarrout);
    unsigned order_shift=shared_order?0:citycount, out_shift=citycount;
    unsigned cities_shift = isEuclidean?citycount*2:citycount*citycount;

    for(unsigned i=0; i<batchsize; i++){
        TSPinstance* tspi;
        if(isEuclidean)
            tspi = new TSPinstanceEuclidean(citycount, cities+cities_shift*i, order+order_shift*i, out+out_shift*i);
        else
            tspi = new TSPinstanceNonEuclidean(citycount, cities+cities_shift*i, order+order_shift*i, out+out_shift*i);
        Insertion* solver = new Insertion(tspi);
        instances.push_back(solver);
    }
    
    return instances;
}

static PyObject *
tsp_insertion_random(PyObject *self, PyObject *args)
{
    /* ----------------- read cities' position from PyObject ----------------- */
    PyObject *pycities, *pyorder, *pyout;
    int isEuclidean = 1;
    if (!PyArg_ParseTuple(args, "OOpO", &pycities, &pyorder, &isEuclidean, &pyout))
        return NULL;
    
    TaskList<TSPInsertion> solvers = read_tsp_instance<TSPInsertion>(pycities, pyorder, isEuclidean, pyout, false);
    if(solvers.size()!=1)
        return NULL;
    
    float distance = solvers.solve_first();
    PyObject *pyresult = PyFloat_FromDouble(distance);
    return pyresult;
}

static PyObject*
tsp_insertion_random_parallel(PyObject *self, PyObject *args)
{
    PyObject *pycities, *pyorder, *pyout;
    int isEuclidean = 1, numThreads_ = 0;
    if (!PyArg_ParseTuple(args, "OOpiO", &pycities, &pyorder, &isEuclidean, &numThreads_, &pyout))
        return NULL;

    TaskList<TSPInsertion> solvers = read_tsp_instance<TSPInsertion>(pycities, pyorder, isEuclidean, pyout, true);
    solvers.solve_parallel(numThreads_);
    return Py_None;
}

#endif