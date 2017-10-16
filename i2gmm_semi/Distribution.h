#pragma once
#include <random>
#include "Vector.h"
using namespace std;
extern default_random_engine generator;
extern uniform_real_distribution<double> distribution;

typedef int dist_t;

/// <summary> The interface for distributions.
/// </summary>
template <class T>
class Distribution
{
public:
	static const dist_t NORMAL = 1;
	static const dist_t STUDENT_T = 2;
	static const dist_t STU_NIW = 5;
	static const dist_t STU_GLOBAL = 3;
	static const dist_t STU_LOCAL = 4;

	static Distribution<T>* generate(dist_t choice);	// Factory method
	static Distribution<T>* generate(dist_t choice, Distribution<T>*);

	virtual ~Distribution() {};

	virtual double likelihood(T& x) = 0; // log likelihood
	virtual T& rnd() = 0;
	virtual void addData(const T& x){};		// update the distribution by obtaining x
	virtual void addData(const T& x, double weight){};
	virtual void removeData(const T& x){};	// update the distribution by removing x
	virtual void removeData(const T& x, double weight){};
	//virtual void update(Statistics* s);	// Update parameters for the distribution.
};

Vector urand(int n);
Vector rand(int n, int max);
Vector rand(int n);
double urand();
