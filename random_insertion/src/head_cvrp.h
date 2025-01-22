#ifndef __RANDOM_INSERTION_CORE_HEAD_CVRP
#define __RANDOM_INSERTION_CORE_HEAD_CVRP

#include "head_common.h"

class CVRPInstance{
public:
friend class CVRPInsertion;
	unsigned citycount;
	CVRPInstance(unsigned cc, float* cp, unsigned* dm, float* dp, unsigned cap, unsigned *outorder, unsigned *outseq, unsigned maxroutecount):
        citycount(cc),citypos(cp),demand(dm),depotpos(dp),capacity(cap),outorder(outorder),outseq(outseq),maxroutecount(maxroutecount){};
	
	float getdistance(unsigned a, unsigned b){
		float* p1 = (a<citycount)?citypos + (a<<1):depotpos;
		float* p2 = (b<citycount)?citypos + (b<<1):depotpos;
		return calc_distance(p1, p2);
	}

private:
	float *citypos;     // nx2
	unsigned *demand;   // n
	float *depotpos;    // 2
	unsigned *outorder; // n
	unsigned *outseq;
	unsigned capacity;
	unsigned maxroutecount;
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
	void randomInsertion(unsigned *order);

private:
	CVRPInstance* cvrpi;
};

#endif