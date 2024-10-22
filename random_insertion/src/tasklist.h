#ifndef __RANDOM_INSERTION_CORE_TASKLIST
#define __RANDOM_INSERTION_CORE_TASKLIST

#include "head.h"

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