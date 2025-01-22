#include "head_cvrp.h"
#include <Python.h>

struct Route{
	Node head;
	unsigned demand;
	float length;
};


void CVRPInsertion::randomInsertion(unsigned *order){
	// initialize ==============================
	const unsigned cc = cvrpi->citycount;
	const unsigned max_routes = cvrpi->maxroutecount;
	const unsigned capacity = cvrpi->capacity;
	unsigned total_routes = 0;
	Node all_nodes[cc];
	Route routes[max_routes];

	// start loop ==================================
	for(unsigned i=0; i<cc; ++i){
		Node *curr = all_nodes+i;
		unsigned currcity = curr->value = order[i];

		float depotdist = cvrpi->getdistance(currcity, cc);
		float mincost = 2.0 * depotdist;
		unsigned currdemand = cvrpi->demand[currcity];
		Route* minroute = nullptr;
		Node* minnode = nullptr;
		
		// get insert posiion with minimum cost
		for(unsigned j = 0; j<total_routes; ++j){
			Route& route = routes[j];
			if(route.demand + currdemand > capacity)
				continue;
			Node *headnode = &(route.head);
			Node *thisnode = headnode, *nextnode;
			float thisdist = depotdist, nextdist;
			do{
				nextnode = thisnode->next;
				nextdist = cvrpi->getdistance(nextnode->value, currcity);
				float delta = thisdist + nextdist - nextnode->length;
				if(delta < mincost)
					mincost = delta, minnode = thisnode, minroute = &route;
				thisnode = nextnode, thisdist = nextdist;
			}while(nextnode!=headnode);
		}

		// update state
		Route* route = nullptr;
		Node* pre = nullptr;
		if(minroute == nullptr){
			Route &new_route = routes[total_routes++];
			pre = new_route.head.next = &new_route.head;
			new_route.head.value = cc;
			new_route.head.length = 0;
			new_route.length = 0.0;
			new_route.demand = 0;
			route = &new_route;
			pre->next->length = curr->length = depotdist;

			mincost = depotdist * 2.0;
		}else{
			pre = minnode, route = minroute;
			curr->length = cvrpi->getdistance(pre->value, currcity);
			pre->next->length = cvrpi->getdistance(currcity, pre->next->value);
		}
		Node* next = pre->next;
		pre->next = curr, curr->next = next;
		route->demand += currdemand;
		route->length += mincost;
	}
	
	// get routes =========================
	unsigned routecount = 0, accu = 0;
	for(unsigned j = 0; j<total_routes; ++j){
		Route& route = routes[j];
		Node* headnode = &(route.head);
		Node* currnode = headnode->next;
		cvrpi->outseq[routecount++] = accu;
		
		while(currnode!=headnode){
			cvrpi->outorder[accu++] = currnode->value;
			currnode = currnode->next;
		}
	}
	cvrpi->outseq[routecount++] = accu;

	return;
}



