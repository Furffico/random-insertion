#ifndef __RANDOM_INSERTION_INTERFACE_TSP
#define __RANDOM_INSERTION_INTERFACE_TSP

#include "head.h"
#include <Python.h>
#include "numpy/arrayobject.h"
#include <thread>

float get_tsp_insertion_result(TSPinstance *tspi, unsigned *order, unsigned *out){
    TSPInsertion ins = TSPInsertion(tspi);
    ins.randomInsertion(order);
    float distance = ins.getResult(out);
    return distance;
}

static PyObject *
tsp_insertion_random(PyObject *self, PyObject *args)
{
    /* ----------------- read cities' position from PyObject ----------------- */
    PyObject *pycities, *pyorder, *pyout;

    int isEuclidean = 1;
    if (!PyArg_ParseTuple(args, "OOpO", &pycities, &pyorder, &isEuclidean, &pyout))
        return NULL;
    if (!PyArray_Check(pycities) || !PyArray_Check(pyorder)|| !PyArray_Check(pyout))
        return NULL;
    
    PyArrayObject *pyarrcities = (PyArrayObject *)pycities, *pyarrorder = (PyArrayObject *)pyorder, *pyarrout = (PyArrayObject *)pyout;

    #ifndef SKIPCHECK
    if (PyArray_NDIM(pyarrcities) != 2 || PyArray_TYPE(pyarrcities) != NPY_FLOAT32
        || PyArray_NDIM(pyarrorder) != 1 || PyArray_TYPE(pyarrorder) != NPY_UINT32
        || PyArray_NDIM(pyarrout) != 1  || PyArray_TYPE(pyarrout) != NPY_UINT32)
        return NULL;
    #endif

    npy_intp *shape = PyArray_SHAPE(pyarrcities);
    unsigned citycount = (unsigned)shape[0];

    #ifndef SKIPCHECK
    if ((unsigned)shape[1]!=(isEuclidean?2:citycount)
        || (unsigned)PyArray_SHAPE(pyarrorder)[0]!=citycount 
        || (unsigned)PyArray_SHAPE(pyarrout)[0]!=citycount)
        return NULL;
    #endif

    float *cities = (float *)PyArray_DATA(pyarrcities);
    unsigned *order = (unsigned *)PyArray_DATA(pyarrorder);
    unsigned *out = (unsigned *)PyArray_DATA(pyarrout);
    
    TSPinstance *tspi;
    if(isEuclidean)
        tspi = new TSPinstanceEuclidean(citycount, cities);
    else
        tspi = new TSPinstanceNonEuclidean(citycount, cities);

    /* ---------------------------- random insertion ---------------------------- */
    float distance = get_tsp_insertion_result(tspi, order, out);

    /* ----------------------- convert output to PyObject ----------------------- */
    PyObject *pyresult = PyFloat_FromDouble(distance);

    delete tspi;
    return pyresult;
}

static PyObject*
tsp_insertion_random_parallel(PyObject *self, PyObject *args)
{
    /* ----------------- read cities' position from PyObject ----------------- */
    PyObject *pycities, *pyorder, *pyout;

    int isEuclidean = 1, numThreads_ = 0;
    if (!PyArg_ParseTuple(args, "OOpiO", &pycities, &pyorder, &isEuclidean, &numThreads_, &pyout))
        return NULL;
    if (!PyArray_Check(pycities) || !PyArray_Check(pyorder)|| !PyArray_Check(pyout))
        return NULL;

    PyArrayObject *pyarrcities = (PyArrayObject *)pycities, *pyarrorder = (PyArrayObject *)pyorder, *pyarrout = (PyArrayObject *)pyout;


    #ifndef SKIPCHECK
    if (PyArray_NDIM(pyarrcities) != 3 || PyArray_TYPE(pyarrcities) != NPY_FLOAT32 
        || PyArray_NDIM(pyarrorder) != 2 && PyArray_NDIM(pyarrorder) != 1 || PyArray_TYPE(pyarrorder) != NPY_UINT32
        || PyArray_NDIM(pyarrout) != 2 || PyArray_TYPE(pyarrout) != NPY_UINT32)
        return NULL;
    #endif
    
    bool shared_order = PyArray_NDIM(pyarrorder) == 1;
    npy_intp *shape = PyArray_SHAPE(pyarrcities);
    unsigned batchsize = (unsigned)shape[0], citycount = (unsigned)shape[1];

    #ifndef SKIPCHECK
    if ((unsigned)shape[2]!=(isEuclidean?2:citycount)
        || shared_order && (unsigned)PyArray_SHAPE(pyarrorder)[0] != citycount 
        || !shared_order && ((unsigned)PyArray_SHAPE(pyarrorder)[0] != batchsize || (unsigned)PyArray_SHAPE(pyarrorder)[1] != citycount)
        || (unsigned)PyArray_SHAPE(pyarrout)[0] != batchsize || (unsigned)PyArray_SHAPE(pyarrout)[1] != citycount)
        return NULL;
    #endif

    float *cities = (float *)PyArray_DATA(pyarrcities);
    unsigned *order = (unsigned *)PyArray_DATA(pyarrorder);
    unsigned *out = (unsigned *)PyArray_DATA(pyarrout);

    /* ---------------------------- setup ---------------------------- */
    unsigned order_shift=shared_order?0:citycount, out_shift=citycount;
    
    TSPinstance **tspi = new TSPinstance*[batchsize];
    if(isEuclidean){
        unsigned cities_shift = citycount*2;
        for(unsigned i=0; i<batchsize; i++)
            tspi[i] = new TSPinstanceEuclidean(citycount, cities+cities_shift*i);
    }else{
        unsigned cities_shift = citycount*citycount;
        for(unsigned i=0; i<batchsize; i++)
            tspi[i] = new TSPinstanceNonEuclidean(citycount, cities+cities_shift*i);
    }

    unsigned numThreads = numThreads_ > 0 ? numThreads_ : std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    numThreads = std::min(numThreads, batchsize);
    
    unsigned chunkSize = batchsize / numThreads;
    if(chunkSize * numThreads != batchsize) chunkSize++;
    
    /* ---------------------------- random insertion ---------------------------- */

    auto function = [tspi, order, out, order_shift, out_shift](unsigned start, unsigned end){
        for (int i=start; i<end; i++){
            get_tsp_insertion_result(tspi[i], order+order_shift*i, out+out_shift*i);
        }
    };

    std::vector<std::thread> threads;
    for (int start=0; start<batchsize; start+=chunkSize){
        unsigned end = std::min(start+chunkSize, batchsize);
        threads.emplace_back(function, start, end);
    }
    for (auto& t: threads) t.join();

    /* ---------------------------- clean up ---------------------------- */
    delete []tspi;
    return Py_None;
}

#endif