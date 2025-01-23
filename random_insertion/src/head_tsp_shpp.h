#ifndef __RANDOM_INSERTION_CORE_HEAD_TSP_SHPP
#define __RANDOM_INSERTION_CORE_HEAD_TSP_SHPP

#include "head_common.h"

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
    ~TSPInsertion(){
        if(tspi)
            delete tspi;
    };
    float solve();

private:
    TSPinstance *tspi;
};

class SHPPInsertion: public InsertionSolver
{
public:
    SHPPInsertion(TSPinstance *tspinstance): instance(tspinstance){};
    ~SHPPInsertion(){
        if(instance)
            delete instance;
    };
    float solve();

private:
    TSPinstance *instance;
};

#endif