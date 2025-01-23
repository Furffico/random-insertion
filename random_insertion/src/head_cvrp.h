#ifndef __RANDOM_INSERTION_CORE_HEAD_CVRP
#define __RANDOM_INSERTION_CORE_HEAD_CVRP

#include "head_common.h"

class Route: public Node{
public:
	unsigned demand = 0;
	float route_length = 0.0;
	Route(): Node(){}
};

class CVRPInstance{
public:
friend class CVRPInsertion;
	unsigned citycount;
	CVRPInstance(unsigned cc, float* cp, unsigned* dm, float* dp, unsigned cap, unsigned* inorder, unsigned *outorder, unsigned *outseq, unsigned maxroutecount):
        citycount(cc),citypos(cp),demand(dm),depotpos(dp),capacity(cap),inorder(inorder),outorder(outorder),outseq(outseq),maxroutecount(maxroutecount){};

	float getdistance(unsigned a, unsigned b){
		float *p1 = (a<citycount)?citypos + (a<<1):depotpos;
		float *p2 = (b<citycount)?citypos + (b<<1):depotpos;
		return calc_distance(p1, p2);
	}

private:
	float *citypos;     // nx2
	unsigned *demand;   // n
	float *depotpos;    // 2
	unsigned *inorder;  // n
	unsigned *outorder; // n
	unsigned *outseq;
	unsigned capacity;
	unsigned maxroutecount;
};

class CVRPInsertion: public InsertionSolver
{
public:
	CVRPInsertion(CVRPInstance* cvrpi):cvrpi(cvrpi){};
	~CVRPInsertion(){
		if(cvrpi)
			delete cvrpi;
	};
	float solve();

private:
	CVRPInstance* cvrpi;
};

#endif