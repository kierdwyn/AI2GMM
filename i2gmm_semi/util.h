#pragma once
#include <math.h>
#include <new>
#include "Vector.h"
#include "Distribution.h"
// From Tom Minka's LightSpeed package
#define mem(X,n) (X*) malloc(sizeof(X)*(n));

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif


double gamln(double x);
double gamlnd(int x, int d);
double harmean(double x,double y);
bool checkVectors(Vector& v1, Vector& v2);

double getGamln(double x);
int sampleFromLog(Vector& v);
int sample(Vector& v);
vector<int> trange(int max, int nparts, int id);

bool fexists(const char *filename);