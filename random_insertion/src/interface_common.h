
#ifndef __RANDOM_INSERTION_CORE_INTERFACE_COMMON
#define __RANDOM_INSERTION_CORE_INTERFACE_COMMON

#include <array>
#include <Python.h>
#include "numpy/arrayobject.h"
#include "head_common.h"

template <typename T, std::size_t dim, enum NPY_TYPES typecode>
T* _check_and_convert(PyObject *pyobj, std::array<unsigned, dim> shape){
    if (!PyArray_Check(pyobj))
        return nullptr;
    PyArrayObject *pyarrcities = (PyArrayObject *)pyobj;
    #ifndef SKIPCHECK
    if (PyArray_NDIM(pyarrcities)!= dim || PyArray_TYPE(pyarrcities)!= typecode)
        return nullptr;
    npy_intp *sh = PyArray_SHAPE(pyarrcities);
    for (unsigned i = 0; i < dim; i++)
        if (shape[i] != sh[i])
            return nullptr;
    #endif
    return (T*)PyArray_DATA(pyarrcities);
}

template <std::size_t dim>
float* check_and_convert_float(PyObject *pyobj, std::array<unsigned, dim> shape){
    return _check_and_convert<float, dim, NPY_FLOAT32>(pyobj, shape);
};

template <std::size_t dim>
unsigned* check_and_convert_unsigned(PyObject *pyobj, std::array<unsigned, dim> shape){
    return _check_and_convert<unsigned, dim, NPY_UINT32>(pyobj, shape);
};

template<class Solver = InsertionSolver>
class TaskList: public std::vector<Solver*>
{
    static_assert(std::is_base_of<InsertionSolver, Solver>::value, "Solver must be a subclass of InsertionSolver");
public:
    float solve_first(){
        unsigned batchsize = this->size();
        if(batchsize==0) return -1;

        Solver* task = this->at(0);
        if(task!=nullptr)
            return task->solve();
        return -1;
    }

    void solve_parallel(unsigned num_threads_=0){
        unsigned batchsize = this->size();
        if(batchsize==0) return;
        else if(batchsize==1){solve_first(); return;}

        unsigned num_threads = num_threads_ > 0 ? num_threads_ : std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        num_threads = std::min(num_threads, batchsize);
        
        unsigned chunkSize = batchsize / num_threads;
        if(chunkSize * num_threads != batchsize) chunkSize++;
        
        /* ---------------------------- random insertion ---------------------------- */
        Solver** tl = this->data();
        auto function = [tl](int start, int end){
            for (int i=start; i<end; i++)
                if(tl[i]!=nullptr)
                    tl[i]->solve();
        };

        std::vector<std::thread> threads;
        for (int start=0; start<(int)batchsize; start+=chunkSize){
            int end = std::min(start+(int)chunkSize, (int)batchsize);
            threads.emplace_back(function, start, end);
        }
        for (auto& t: threads) t.join();
    }
    ~TaskList(){
        Solver** tl = this->data();
        for(unsigned i=0;i<this->size();i++){
            delete tl[i];
            tl[i]=nullptr;
        }
    }
};

#endif