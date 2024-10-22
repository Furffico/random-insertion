#include "head.h"


void SHPPInsertion::initState()
{
    unsigned* order = instance -> order;
    unsigned cc = instance -> citycount;
    Node dummynode, *lastnode = &dummynode;

    unsigned subcc=cc-1;
    Node *headnode = new Node(0);
    Node *tailnode = (route=headnode)->next = new Node(subcc);
    // The distance from node A to node B is stored in node B
    tailnode->length = instance->getdist(headnode->value, tailnode->value);

    for (unsigned i = 0; i < cc; ++i)
    {
        unsigned index=order[i];
        if(index!=0 && index != subcc)
            lastnode = (lastnode->next = new Node(index));
    }
    vacant = dummynode.next;
    dummynode.next = nullptr;
}

Node *SHPPInsertion::getVacantNode()
{
    Node *result = vacant;
    if (vacant != nullptr)
        vacant = vacant->next, result->next = nullptr;
    return result;
}

void SHPPInsertion::randomInsertion()
{
    initState();

    Node *curr, *next;
    while ((curr=getVacantNode())!=nullptr){
        unsigned city = curr->value;
        // get target list and distances
        // and get insert position with minimum cost
        Node *thisnode, *nextnode=route, *minnode = route;
        float thisdist, nextdist;
        float mindelta = INFINITY, td=0.0, nd=0.0;

        while((nextnode=(thisnode=nextnode)->next)!=nullptr){
            thisdist = instance->getdist(thisnode->value, city);
            nextdist = instance->getdist(city, nextnode->value);
            float delta = thisdist + nextdist - nextnode->length;
            if (delta < mindelta)
                mindelta = delta, minnode = thisnode, td = thisdist, nd = nextdist;
        }
        // insert the selected node
        next = minnode->next;
        minnode->next = curr;
        curr->next = next;
        curr->length = td, next->length = nd;
    }
}


float SHPPInsertion::getResult(unsigned* output){
    if(output==nullptr || route == nullptr)
        return -1.0;
    unsigned cc = instance->citycount;

    // get node order
    Node *node = route;
    float distance = 0.0;
    for (unsigned i = 0; i < cc; i++)
    {
        output[i] = node->value;
        distance += node->length;
        node = node->next;
    }
    return distance;
}

SHPPInsertion::~SHPPInsertion(){
    if(route!=nullptr){
        Node* last, *node = route;
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
    if(instance!=nullptr)
        delete instance;
}

