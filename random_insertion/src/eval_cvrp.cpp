#include "head_cvrp.h"
#include <Python.h>


float CVRPInsertion::solve(){
	// initialize ==============================
	const unsigned cc = cvrpi->citycount;
	const unsigned max_routes = cvrpi->maxroutecount;
	const unsigned capacity = cvrpi->capacity;
	const unsigned *order = cvrpi->inorder;
	unsigned total_routes = 0;
	std::vector<Node> all_nodes(cc);
	std::vector<Route> routes(max_routes);

	// start loop ==================================
	for(unsigned i=0; i<cc; ++i){
		Node &curr = all_nodes[i];
		const unsigned currcity = curr.value = order[i];
		const unsigned currdemand = cvrpi->demand[currcity];
		const float depotdist = cvrpi->getdistance(currcity, cc);
		float mincost = 2.0 * depotdist;
		Route *minroute = nullptr;
		Node *minnode = nullptr;
		
		// get insert posiion with minimum cost
		for(unsigned j = 0; j<total_routes; ++j){
			Route &route = routes[j];
			if(route.demand + currdemand > capacity)
				continue;
			Node *headnode = &route, *thisnode = headnode, *nextnode;
			float thisdist = depotdist, nextdist;
			do{
				nextnode = thisnode->next;
				nextdist = cvrpi->getdistance(nextnode->value, currcity);
				float delta = thisdist + nextdist - nextnode->length;
				if(delta <= mincost)
					mincost = delta, minnode = thisnode, minroute = &route;
				thisnode = nextnode, thisdist = nextdist;
			}while(nextnode!=headnode);
		}

		// update state
		if(minroute == nullptr){
			Route &route = routes[total_routes++];
			route.value = cc;
			route.demand = currdemand;
			route.route_length = depotdist * 2.0;
			curr.length = route.length = depotdist;
			route.next = &curr, curr.next = (Node*)&route;
		}else{
			Node *next = minnode->next;
			curr.length = cvrpi->getdistance(minnode->value, currcity);
			next->length = cvrpi->getdistance(currcity, next->value);
			minnode->next = &curr, curr.next = next;
			minroute->demand += currdemand;
			minroute->route_length += mincost;
		}
	}
	
	// get routes =========================
	unsigned routecount = 0, accu = 0;
	float total_length = 0.0;
	for(unsigned j = 0; j<total_routes; ++j){
		Route &route = routes[j];
		Node *headnode = (Node*)&route, *currnode = route.next;
		cvrpi->outseq[routecount++] = accu;
		total_length += route.route_length;
		
		while(currnode!=headnode){
			cvrpi->outorder[accu++] = currnode->value;
			currnode = currnode->next;
		}
	}
	cvrpi->outseq[routecount++] = accu;

	return total_length;
}



