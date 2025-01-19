#ifndef __RANDOM_INSERTION_CORE_HEAD_CVRP
#define __RANDOM_INSERTION_CORE_HEAD_CVRP

#include "head_common.h"

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