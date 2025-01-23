#include "head_tsp_shpp.h"

float SHPPInsertion::solve(){
    const unsigned cc = instance->citycount;
    const unsigned lastcity = cc-1;
    const unsigned *order = instance->order;
    Node nodes[cc];
    Node *head = &nodes[0], *tail=&nodes[lastcity];
    head->next = tail, tail->next = nullptr;
    head->value = 0, tail->value = lastcity;
    tail->length = instance->getdist(head->value, tail->value);

    for (unsigned i = 0; i < cc; ++i){
        if(order[i]==0 || order[i]==lastcity)
            continue;

        Node &curr = nodes[i];
        unsigned city = curr.value = order[i];

        // determine the insert position with minimum cost
        Node *thisnode = head, *nextnode = head->next, *minnode = head;
        float mindelta = INFINITY, td=0.0, nd=0.0;
        do{
            float thisdist = instance->getdist(thisnode->value, city);
            float nextdist = instance->getdist(city, nextnode->value);
            float delta = thisdist + nextdist - nextnode->length;
            if (delta < mindelta)
                mindelta = delta, minnode = thisnode, td = thisdist, nd = nextdist;
            thisnode = nextnode;
            nextnode = nextnode->next;
        }while(nextnode!=nullptr);

        // insert the selected node
        curr.next = minnode->next;
        minnode->next = &curr;
        curr.length = td, curr.next->length = nd;
    }

    // get node order
    Node *node = head;
    float distance = 0.0;
    for (unsigned i = 0; i < cc; ++i)
    {
        instance->out[i] = node->value;
        distance += node->length;
        node = node->next;
    }
    return distance;
}

