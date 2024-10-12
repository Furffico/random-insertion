#ifndef __RANDOM_INSERTION_CORE_HEAD
#define __RANDOM_INSERTION_CORE_HEAD

#include <vector>
#include <thread>
#include <math.h>
#include <type_traits>

inline float calc_distance(float* a, float* b){
	float d1 = *a - *b, d2 = *(a + 1) - *(b + 1);
	return sqrtf32(d1*d1+d2*d2);
}

class Node
{
public:
    Node *next = nullptr;
    unsigned value = 0;
    float length = 0;
};

class InsertionSolver
{
public:
    InsertionSolver(){};
    virtual float solve(){return 0.0f;};
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
        auto function = [tl](unsigned start, unsigned end){
            for (int i=start; i<end; i++)
                if(tl[i]!=nullptr)
                    tl[i]->solve();
        };

        std::vector<std::thread> threads;
        for (int start=0; start<batchsize; start+=chunkSize){
            unsigned end = std::min(start+chunkSize, batchsize);
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
class TSPinstance
{
public:
    friend class TSPInsertion;
    unsigned citycount;
    // TSPinstance(unsigned cc):citycount(cc){};
    TSPinstance(unsigned cc, unsigned* order, unsigned* out):citycount(cc),order(order),out(out){};
    virtual float getdist(unsigned cityA, unsigned cityB){
        return 0.0f;
    };
private:
    unsigned* order=nullptr;
    unsigned* out=nullptr;
};

class TSPinstanceEuclidean: public TSPinstance
{
public:
    TSPinstanceEuclidean(unsigned cc, float *cp, unsigned* order, unsigned* out): TSPinstance(cc,order,out), citypos(cp){};
    float getdist(unsigned a, unsigned b){
        float *p1 = citypos + (a << 1), *p2 = citypos + (b << 1);
        float d1 = *p1 - *p2, d2 = *(p1 + 1) - *(p2 + 1);
        return sqrtf32(d1 * d1 + d2 * d2);
    };
    ~TSPinstanceEuclidean(){
        citypos = nullptr;
    };
private:
    float *citypos;
};

class TSPinstanceNonEuclidean: public TSPinstance
{
public:
    TSPinstanceNonEuclidean(unsigned cc, float *distmat, unsigned* order, unsigned* out): TSPinstance(cc,order,out), distmat(distmat){};
    float getdist(unsigned a, unsigned b){
        return distmat[citycount * a + b];
    };
    ~TSPinstanceNonEuclidean(){
        distmat = nullptr;
    };
private:
    float *distmat;
};

class TSPInsertion: public InsertionSolver
{
public:
    TSPInsertion(TSPinstance *tspinstance): tspi(tspinstance){};
    ~TSPInsertion();
    float solve(){
        randomInsertion(tspi->order);
        float distance = getResult(tspi->out);
        return distance;
    }

private:
    TSPinstance *tspi;
    Node *vacant = nullptr;
    Node *route = nullptr;
    Node *getVacantNode();
    void initState(unsigned *order);
    void randomInsertion(unsigned *order);
    float getResult(unsigned* output);
};

class CVRPInstance{
public:
	unsigned citycount;
	float *citypos;   // nx2
	unsigned *demand; // n
	float *depotpos;  // 2
	unsigned capacity;
	CVRPInstance(unsigned cc, float* cp, unsigned* dm, float* dp, unsigned cap):
        citycount(cc),citypos(cp),demand(dm),depotpos(dp),capacity(cap){};
	float getdistance(unsigned a, unsigned b){
		float* p1 = (a<citycount)?citypos + (a<<1):depotpos;
		float* p2 = (b<citycount)?citypos + (b<<1):depotpos;
		return calc_distance(p1, p2);
	}
};

struct CVRPReturn{
	unsigned routes;
	unsigned* order;
	unsigned* routesep;
};

class CVRPInsertion
{
public:
	CVRPInsertion(CVRPInstance* cvrpi):cvrpi(cvrpi){};
	CVRPReturn *randomInsertion(unsigned *order, float exploration);

private:
	CVRPInstance* cvrpi;
};

struct Route{
	Node* head;
	unsigned demand;
	float length;
};

#endif