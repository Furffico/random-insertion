#include "head.h"

void TSPInsertion::initState(unsigned *order)
{
    unsigned cc = tspi->citycount;
    Node dummynode;
    Node *lastnode = &dummynode;
    for (unsigned i = 0; i < cc; i++)
        lastnode = (lastnode->next = new Node(order[i]));
    vacant = dummynode.next;
    dummynode.next = nullptr;
}


Node *TSPInsertion::getVacantNode()
{
    Node *result = vacant;
    if (vacant != nullptr)
    {
        vacant = vacant->next;
        result->next = nullptr;
    }
    return result;
}

void TSPInsertion::randomInsertion(unsigned *order)
{
    initState(order);
    unsigned cc = tspi->citycount;
    // generate initial route with 2 nodes
    {
        Node *node1 = getVacantNode();
        Node *node2 = getVacantNode();
        route = node2->next = node1;
        node1->next = node2;
        node2->length = tspi->getdist(node1->value, node2->value);
        node1->length = tspi->getdist(node2->value, node1->value);
    }

    for (unsigned i = 2; i < cc; i++)
    {
        // get a city from vacant
        Node *curr = getVacantNode();
        unsigned city = curr->value;

        // get target list and distances
        // and get insert position with minimum cost
        Node *thisnode = route, *nextnode = thisnode->next;
        float thisdist = 0, nextdist = 0;
        Node *minnode = thisnode;
        float mindelta = INFINITY;
        float td = 0.0, nd = 0.0;

        for (unsigned j = 0; j < i; j++)
        {
            nextnode = thisnode->next;
            thisdist = tspi->getdist(thisnode->value, city);
            nextdist = tspi->getdist(city, nextnode->value);
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
        pre->next = curr, curr->next = next;
        curr->length = td, next->length = nd;
    }
}

float TSPInsertion::getResult(unsigned* output){
    if(output==nullptr || route == nullptr)
        return -1.0;

    // get node order
    Node *node = route;
    float distance = 0.0;
    for (unsigned i = 0; i < tspi->citycount; i++)
    {
        output[i] = node->value;
        distance += node->length;
        node = node->next;
    }
    return distance;
}

TSPInsertion::~TSPInsertion(){
    if(route!=nullptr){
        Node* last, *node = route->next;
        route->next = nullptr;
        while(node!=nullptr){
            node = (last = node)->next;
            delete last;
        }
        route = node = last = nullptr;
    }
    if(vacant!=nullptr){
        Node* last, *node = vacant;
        while(node!=nullptr){
            node = (last = node)->next;
            delete last;
        }
        vacant = nullptr;
    }
    if(tspi!=nullptr)
        delete tspi;
}

