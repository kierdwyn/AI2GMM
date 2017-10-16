#include "Distribution.h"
#include "Normal.h"
#include "Stut.h"
#include "Stut_i3gmm.h"

default_random_engine generator(time(NULL));
uniform_real_distribution<double> distribution(0.0, 1.0);

template <class T>
Distribution<T>* Distribution<T>::generate(dist_t choice){
	Distribution<T> *dist;
	switch (choice)
	{
	case Distribution<T>::NORMAL:
		dist = new Normal();
		break;
	case Distribution<T>::STUDENT_T:
		dist = new Stut();
		break;
	case Distribution<T>::STU_NIW:
		dist = new StutNIW();
		break;
	case Distribution<T>::STU_GLOBAL:
		dist = new StutGlobal3();
		break;
	case Distribution<T>::STU_LOCAL:
		dist = new StutLocal3();
		break;
	default:
		dist = new Normal();
		break;
	}
	return dist;
}

template <class T>
Distribution<T>* Distribution<T>::generate(dist_t choice, Distribution<T>* copy){
	Distribution<T> *dist = nullptr;
	switch (choice)
	{
	case Distribution<T>::NORMAL:
		dist = new Normal(*dynamic_cast<Normal *>(copy));
		break;
	case Distribution<T>::STUDENT_T:
		dist = new Stut(*dynamic_cast<Stut *>(copy));
		break;
	case Distribution<T>::STU_NIW:
		dist = new StutNIW(*dynamic_cast<StutNIW *>(copy));
		break;
	case Distribution<T>::STU_GLOBAL:
		dist = new StutGlobal3(*dynamic_cast<StutGlobal3 *>(copy));
		break;
	case Distribution<T>::STU_LOCAL:
		if (dynamic_cast<StutLocal3*>(copy) != nullptr)
			dist = new StutLocal3(*dynamic_cast<StutLocal3 *>(copy));
		else if (dynamic_cast<StutGlobal3*>(copy) != nullptr)
			dist = new StutLocal3(dynamic_cast<StutGlobal3 *>(copy));
		break;
	default:
		dist = new Normal(*dynamic_cast<Normal *>(copy));
		break;
	}

	if (dist == nullptr){
		printf("No suitable distribution...\n");
		throw;
	}
	return dist;
}

double urand()
{
	return distribution(generator);
}

Vector urand(int n)
{
	Vector v(n);
	for (auto i = 0; i < n; i++)
		v[i] = urand();

	return v;
}

Vector rand(int n, int max)
{
	Vector v(n);
	for (auto i = 0; i < n; i++)
		v[i] = rand() % max;

	return v;
}


Vector rand(int n)
{
	Vector rv(n);
	for (auto i = 0; i < n; i++)
		rv[i] = urand();

	return rv;
}

template class Distribution < Vector > ;