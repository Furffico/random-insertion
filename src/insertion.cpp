#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <thread>
#include <math.h>
#include <vector>
#include <Python.h>
#include "head.h"
#include "numpy/arrayobject.h"


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
    PyObject *pycities;
    PyObject *pyorder;
    PyObject *pyout;

    int isEuclidean = 1;
    if (!PyArg_ParseTuple(args, "OOpO", &pycities, &pyorder, &isEuclidean, &pyout))
        return NULL;
    if (!PyArray_Check(pycities) || !PyArray_Check(pyorder)|| !PyArray_Check(pyout))
        return NULL;
    
    PyArrayObject *pyarrcities = (PyArrayObject *)pycities;
    PyArrayObject *pyarrorder = (PyArrayObject *)pyorder;
    PyArrayObject *pyarrout = (PyArrayObject *)pyout;

    if (PyArray_NDIM(pyarrcities) != 2 || PyArray_NDIM(pyarrorder) != 1 || PyArray_NDIM(pyarrout) != 1)
        return NULL;
    if (PyArray_TYPE(pyarrcities) != NPY_FLOAT32 || PyArray_TYPE(pyarrorder) != NPY_UINT32 || PyArray_TYPE(pyarrout) != NPY_UINT32)
        return NULL;

    npy_intp *shape = PyArray_SHAPE(pyarrcities);
    unsigned citycount = (unsigned)shape[0];
    if ((unsigned)PyArray_SHAPE(pyarrorder)[0]!=citycount || (unsigned)PyArray_SHAPE(pyarrout)[0]!=citycount)
        return NULL;

    float *cities = (float *)PyArray_DATA(pyarrcities);
    unsigned *order = (unsigned *)PyArray_DATA(pyarrorder);
    unsigned *out = (unsigned *)PyArray_DATA(pyarrout);
    
    TSPinstance *tspi;
    if(isEuclidean){
        if ((unsigned)shape[1]!=2) return NULL;
        tspi = new TSPinstanceEuclidean(citycount, cities);
    }else{
        if ((unsigned)shape[1]!=citycount) return NULL;
        tspi = new TSPinstanceNonEuclidean(citycount, cities);
    }

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
    PyObject *pycities;
    PyObject *pyorder;
    PyObject *pyout;

    int isEuclidean = 1;
    if (!PyArg_ParseTuple(args, "OOpO", &pycities, &pyorder, &isEuclidean, &pyout))
        return NULL;
    if (!PyArray_Check(pycities) || !PyArray_Check(pyorder)|| !PyArray_Check(pyout))
        return NULL;
    
    PyArrayObject *pyarrcities = (PyArrayObject *)pycities;
    PyArrayObject *pyarrorder = (PyArrayObject *)pyorder;
    PyArrayObject *pyarrout = (PyArrayObject *)pyout;

    if (PyArray_NDIM(pyarrcities) != 3 || PyArray_NDIM(pyarrorder) != 2 && PyArray_NDIM(pyarrorder) != 1 || PyArray_NDIM(pyarrout) != 2)
        return NULL;
    if (PyArray_TYPE(pyarrcities) != NPY_FLOAT32 || PyArray_TYPE(pyarrorder) != NPY_UINT32 || PyArray_TYPE(pyarrout) != NPY_UINT32)
        return NULL;
    
    bool shared_order = PyArray_NDIM(pyarrorder) == 1;

    npy_intp *shape = PyArray_SHAPE(pyarrcities);
    unsigned batchsize = (unsigned)shape[0], citycount = (unsigned)shape[1];
    if (shared_order
        ? (unsigned)PyArray_SHAPE(pyarrorder)[0] != citycount
        : (unsigned)PyArray_SHAPE(pyarrorder)[0] != batchsize || (unsigned)PyArray_SHAPE(pyarrorder)[1] != citycount
    )
        return NULL;
    if ((unsigned)PyArray_SHAPE(pyarrout)[0] != batchsize || (unsigned)PyArray_SHAPE(pyarrout)[1] != citycount)
        return NULL;

    float *cities = (float *)PyArray_DATA(pyarrcities);
    unsigned *order = (unsigned *)PyArray_DATA(pyarrorder);
    unsigned *out = (unsigned *)PyArray_DATA(pyarrout);


    /* ---------------------------- setup ---------------------------- */
    unsigned order_shift=shared_order?0:citycount, out_shift=citycount;
    
    TSPinstance **tspi = new TSPinstance*[batchsize];
    if(isEuclidean){
        if ((unsigned)shape[2]!=2) return NULL;
        unsigned cities_shift = citycount*2;
        for(unsigned i=0; i<batchsize; i++)
            tspi[i] = new TSPinstanceEuclidean(citycount, cities+cities_shift*i);
    }else{
        if ((unsigned)shape[2]!=citycount) return NULL;
        unsigned cities_shift = citycount*citycount;
        for(unsigned i=0; i<batchsize; i++)
            tspi[i] = new TSPinstanceNonEuclidean(citycount, cities+cities_shift*i);
    }

    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // 默认使用4个线程
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
    
    PyArrayObject *pyarrcities = (PyArrayObject *)pycities;
    PyArrayObject *pyarrorder = (PyArrayObject *)pyorder;
    PyArrayObject *pyarrdemands = (PyArrayObject *)pydemands;

    if (PyArray_NDIM(pyarrcities) != 2 || PyArray_NDIM(pyarrorder) != 1 || PyArray_NDIM(pyarrdemands) != 1)
        return NULL;
    if (PyArray_TYPE(pyarrcities) != NPY_FLOAT32 || PyArray_TYPE(pyarrorder) != NPY_UINT32 || PyArray_TYPE(pyarrdemands) != NPY_UINT32 )
        return NULL;

    // std::printf("running\n");

    npy_intp *shape = PyArray_SHAPE(pyarrcities);
    unsigned citycount = (unsigned)shape[0];
    float *cities = (float *)PyArray_DATA(pyarrcities);
    unsigned *order = (unsigned *)PyArray_DATA(pyarrorder);
    unsigned *demands = (unsigned *)PyArray_DATA(pyarrdemands);
    float depotpos[2] = {depotx, depoty};

    /* ---------------------------- random insertion ---------------------------- */
    CVRPInstance cvrpi = CVRPInstance(citycount, cities, demands, depotpos, capacity);
    CVRPInsertion ins = CVRPInsertion(&cvrpi);

    // std::printf("insertion\n");
    CVRPReturn *result = ins.randomInsertion(order, exploration);
    // std::printf("insertion finished\n");
    /* ----------------------- convert output to PyObject ----------------------- */
    npy_intp dims = citycount, dims2 = result->routes;
    PyObject *returntuple = PyTuple_Pack(2, 
        PyArray_SimpleNewFromData(1, &dims, NPY_UINT32, result->order),
        PyArray_SimpleNewFromData(1, &dims2, NPY_UINT32, result->routesep)
    );

    return returntuple;
}

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
