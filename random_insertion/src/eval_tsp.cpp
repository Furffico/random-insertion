#include "head_tsp_shpp.h"


float TSPInsertion::solve(){
    // initialize nodes
    const unsigned cc = tspi->citycount;
    const unsigned *order = tspi->order;
    std::vector<Node> nodes(cc);
    
    // generate initial route with 2 nodes
    {
        Node *node1 = &nodes[0], *node2 = &nodes[1];
        node2->next = node1, node1->next = node2;
        node2->length = tspi->getdist(node1->value = order[0], node2->value = order[1]);
        node1->length = tspi->getdist(node2->value, node1->value);
    }

    for (unsigned i = 2; i < cc; ++i){
        // get a city from vacant
        Node &curr = nodes[i];
        const unsigned city = curr.value = order[i];

        // get target list and distances
        // and get insert position with minimum cost
        Node *thisnode = &nodes[0], *nextnode = thisnode->next, *minnode = thisnode;
        float mindelta = INFINITY, td = 0.0, nd = 0.0;

        for (unsigned j = 0; j < i; ++j){
            nextnode = thisnode->next;
            float thisdist = tspi->getdist(thisnode->value, city);
            float nextdist = tspi->getdist(city, nextnode->value);
            float delta = thisdist + nextdist - nextnode->length;
            if (delta < mindelta)
            {
                mindelta = delta, minnode = thisnode;
                td = thisdist, nd = nextdist;
            }
            thisnode = nextnode;
        }

        // insert the selected node
        Node *pre = minnode, *next = minnode->next;
        pre->next = &curr, curr.next = next;
        curr.length = td, next->length = nd;
    }

    // get node order
    Node *node = &nodes[0];
    float distance = 0.0;
    for (unsigned i = 0; i < cc; ++i)
    {
        tspi->out[i] = node->value;
        distance += node->length;
        node = node->next;
    }
    return distance;
}
