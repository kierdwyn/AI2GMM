#pragma once
#include "Stut.h"
#include "Matrix.h"

template <class T> class Table;
class StutLocal3;


class HyperParams {
public:
	// Hidden variables for i3gmm, hyper-params for i2gmm
	static Vector mu0,mu0ML;
	static Matrix psi0,psi0ML;
	static double kappa0, kappa1;
	static int m; // Degree of freedom
	static Matrix last_psi0;
	static double last_kappa0, last_kappa1;
	static int last_m; // Degree of freedom

	HyperParams(){}
	HyperParams(Vector mu0, Matrix sigma0, double m,
		double kappa0, double kappa1);
	HyperParams(Vector mu0, Matrix sigma0, double m,
		double c1, double c2, double alpha0, double beta0,
		double alpha1, double beta1);

	static void init(Vector mu0, Matrix sigma0, double m,
		double kappa0, double kappa1);
	static void init(Vector mu0, Matrix sigma0, double m,
		double c1, double c2, double alpha0, double beta0,
		double alpha1, double beta1);
	template<class Iter> static void update_global_stats(Iter first, Iter last);
	template <class Iter> static void renew_hyper_params(Iter first, Iter last, int psi0choice);

	static Vector mx_all;
	static Matrix sx_all;
	static double nx_all;
	static double n_classes;

private:
	// Hyper-parameters for i3gmm
	static Vector mu00;
	static Matrix sigma0_inv, sigma0;
	static double c1, c2, alpha0, alpha1, beta0, beta1, kap1over0;
};

class StutGlobal3 :public Stut, public HyperParams {
public:
	// Sample statistics
	double n_clusters = 0;
	double sum_weighted_n = 0;	/// \f[ \sum_{l:c_l=k} \frac{n_{kl} \kappa_1}{n_{kl} + \kappa_1} \f]
	double sum_n = 0;
	Vector weighted_mean;	/// \f[ \sum_{l:cl=k} \frac{n_{kl} \kappa_1}{n_{kl} + \kappa_1} * \bar{x}_{kl} \f]
	Matrix sum_scatter;

	// Derived terms
	double kappa_s;
	//Vector mu_k=mu_s=mu;
	//double v_s = eta;
	//Matrix sigma_s;

	// Hidden variables for calculating hyper-parameters
	double n_k = 0, sum_n_kl = 0;
	Matrix sum_scatter_mu_kl;
	Matrix sum_scatter_kl;
	Vector sum_mu_kl;
	Vector mu_h;
	Matrix sigma_h;

	StutGlobal3();
	StutGlobal3(Vector mu0, Matrix psi0, double m,
		double kappa0, double kappa1);
	StutGlobal3(Vector mu0, Matrix sigma0, double m,
		double c1, double c2, double alpha0, double beta0,
		double alpha1, double beta1);

	double loglike(const Table<Vector> &x, int tablelike = 0);
	void add_component(StutLocal3 *x_dist);		// update the distribution by obtaining x
	void remove_component(StutLocal3 *x_dist);	// update the distribution by removing x
	template <class Iter> void add_all(Iter first, Iter last);
	void reset();
	void update_stut(int isML=0);
	void update_hidden_vars();
	void update_hidden_vars(StutLocal3 *x_dist, int sign);

private:
	void update_statistics(StutLocal3 *x_dist, int sign);
	void init();
};

class Stut3 : public Stut, public HyperParams {
public:
	int n_points_unweighted = 0;
	double n_points = 0;
	Vector mean;
	Matrix scatter;

	Stut3();
	virtual ~Stut3() {};

	virtual void addData(const Vector& x, double weight = 1) override;		// update the distribution by obtaining x
	virtual void removeData(const Vector& x, double weight = 1) override;	// update the distribution by removing x
	template <class Iter> void add_all(Iter first, Iter last);
	void reset();

	virtual void update_stut(const Vector* x) = 0;
};

class StutLocal3 : public Stut3 {
public:
	StutGlobal3 *g_dist = nullptr;
	Vector mu_kl; // Hidden variable
	Matrix sigma_s;

	StutLocal3() :Stut3() { update_stut(); }
	StutLocal3(StutGlobal3 *stut);

	void update_hidden_vars();
	void change_global(StutGlobal3 *g) { g_dist = g; update_stut(nullptr); update_hidden_vars();}
	//template <class Iter> void add_all(Iter first, Iter last);

	virtual void update_stut(const Vector* x = nullptr) override;
};

class StutNIW : public Stut3 {
public:
	StutNIW() :Stut3() { update_stut(); }
	virtual void update_stut(const Vector* x = nullptr) override;
};
