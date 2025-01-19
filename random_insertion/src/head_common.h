#ifndef __RANDOM_INSERTION_CORE_HEAD_COMMON
#define __RANDOM_INSERTION_CORE_HEAD_COMMON

#include <vector>
#include <thread>
#include <math.h>
#include <type_traits>

inline float calc_distance(float* a, float* b){
	float d1 = *a - *b, d2 = *(a + 1) - *(b + 1);
	return sqrt(d1*d1+d2*d2);
}

class Node
{
public:
    Node *next = nullptr;
    unsigned value = 0;
    float length = 0;
    Node(){};
    Node(unsigned value):value(value){};
};

class InsertionSolver
{
public:
    InsertionSolver(){};
    virtual float solve(){return 0.0f;};
};

#endif