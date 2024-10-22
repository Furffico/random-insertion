#ifndef __RANDOM_INSERTION_CORE_HEAD
#define __RANDOM_INSERTION_CORE_HEAD

#include <vector>
#include <thread>
#include <math.h>
#include <type_traits>

inline float calc_distance(float* a, float* b){
	float d1 = *a - *b, d2 = *(a + 1) - *(b + 1);
	return sqrt(d1*d1+d2*d2);
}

class Node
{
public:
    Node *next = nullptr;
    unsigned value = 0;
    float length = 0;
    Node(){};
    Node(unsigned value):value(value){};
};

class InsertionSolver
{
public:
    InsertionSolver(){};
    virtual float solve(){return 0.0f;};
};

class TSPinstance
{
public:
    friend class TSPInsertion;
    friend class SHPPInsertion;
    unsigned citycount;
    // TSPinstance(unsigned cc):citycount(cc){};
    TSPinstance(unsigned cc, unsigned* order, unsigned* out):citycount(cc),order(order),out(out){};
    virtual float getdist(unsigned cityA, unsigned cityB){
        return 0.0f;
    };
    virtual ~TSPinstance(){};
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
        return sqrt(d1 * d1 + d2 * d2);
    };
    virtual ~TSPinstanceEuclidean(){ citypos = nullptr; };
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
    virtual ~TSPinstanceNonEuclidean(){ distmat = nullptr; };
private:
    float *distmat;
};

class TSPInsertion: public InsertionSolver
{
public:
    TSPInsertion(TSPinstance *tspinstance): tspi(tspinstance){};
    virtual ~TSPInsertion();
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

class SHPPInsertion: public InsertionSolver
{
public:
    SHPPInsertion(TSPinstance *tspinstance): instance(tspinstance){};
    virtual ~SHPPInsertion();
    float solve(){
        randomInsertion();
        float distance = getResult(instance->out);
        return distance;
    }

private:
    TSPinstance *instance;
    Node *vacant = nullptr;
    Node *route = nullptr;
    Node *getVacantNode();
    void initState();
    void randomInsertion();
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