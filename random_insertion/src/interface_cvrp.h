#ifndef __RANDOM_INSERTION_INTERFACE_CVRP
#define __RANDOM_INSERTION_INTERFACE_CVRP

#include "head_cvrp.h"
#include "interface_common.h"

static PyObject*
cvrp_insertion_random_parallel(PyObject *self, PyObject *args)
{
    /* ----------------- read cities' position from PyObject ----------------- */
    PyObject *pycities, *pyorder, *pydemands, *pyoutorder, *pyoutsep, *pydepots, *pycapacities;
    int num_threads = 0;
    bool shared_capacity = false, shared_order = false;

    // positions depotx depoty demands capacity order
    if (!PyArg_ParseTuple(args, "OOOOOiOO", &pycities, &pydepots, &pydemands, &pycapacities, &pyorder, &num_threads, &pyoutorder, &pyoutsep))
        return NULL;

    // get hyper parameters
    PyArrayObject *arr_cities = (PyArrayObject *)pycities, *arr_outsep = (PyArrayObject *)pyoutsep;
    if (!PyArray_Check(pycities) || PyArray_NDIM(arr_cities) != 3 
        || !PyArray_Check(pyoutsep) || PyArray_NDIM(arr_outsep) != 2)
        return NULL;
    npy_intp *shape = PyArray_SHAPE(arr_cities);
    if(shape[2]!=2)
        return NULL;
    const unsigned batchsize = (unsigned)shape[0], citycount = (unsigned)shape[1];
    const unsigned maxroutecount = (unsigned)PyArray_SHAPE(arr_outsep)[1];

    // check and convert numpy arrays
    float *cities = (float*)PyArray_DATA(arr_cities);
    unsigned *outsep = (unsigned*)PyArray_DATA(arr_outsep);
    float *depotpos = check_and_convert_float(pydepots, std::array<unsigned,2>{batchsize, 2});
    unsigned *demands = check_and_convert_unsigned(pydemands, std::array<unsigned,2>{batchsize, citycount});
    unsigned *outorder = check_and_convert_unsigned(pyoutorder, std::array<unsigned,2>{batchsize, citycount});
    if(cities == nullptr || depotpos==nullptr || demands==nullptr || outorder==nullptr || outsep==nullptr)
        return NULL;
    
    unsigned *capacity = check_and_convert_unsigned(pycapacities, std::array<unsigned,1>{batchsize}), capacity_val;
    if(capacity==nullptr){
        const long value = PyLong_AsLong(pycapacities);
        if(value<0) return NULL;
        capacity_val = (unsigned)value;
        shared_capacity = true;
        capacity = &capacity_val;
    }

    unsigned *order = check_and_convert_unsigned(pyorder, std::array<unsigned,2>{batchsize, citycount});
    if(order==nullptr){
        order = check_and_convert_unsigned(pyorder, std::array<unsigned,1>{citycount});
        if(order==nullptr) return NULL;
        shared_order = true;
    }

    // build tasklist
    TaskList<CVRPInsertion> tasklist({});
    tasklist.reserve(batchsize);

    for(unsigned i=0; i<batchsize; i++){
        CVRPInstance *cvrpi = new CVRPInstance(
            citycount,
            cities+i*citycount*2,
            demands+i*citycount,
            depotpos+i*2,
            shared_capacity?*capacity:*(capacity+i),
            shared_order?order:order+i*citycount,
            outorder+i*citycount,
            outsep+i*maxroutecount,
            maxroutecount);
        CVRPInsertion *ins = new CVRPInsertion(cvrpi);
        tasklist.push_back(ins);
    }

    // perform insertion
    tasklist.solve_parallel(num_threads);

    return Py_None;
}
#endif