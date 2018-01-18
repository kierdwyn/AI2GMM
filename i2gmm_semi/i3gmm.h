#pragma once
#include <unordered_set>
#include <unordered_map>
#include "DP.h"
#include "Matrix.h"
#include "Stut_i3gmm.h"


class I3gmm{
public:
	vector<Customer<Vector>*> allcusts;
	vector<DP<Vector>*> best_clusters;
	vector<Table<Vector>*> best_components;
	vector<Customer<Vector>*> best_customers;
	Matrix sample_labels;
	double* logLikeHistory = nullptr;
	double maxLogLike = -INFINITY;
	int best_sweep = 0;

	bool adjustable = false;
	int outlier_ratio = 0, prior_type = 0; // 0:dp, 1:log2, 2:uniform
	bool tune_on_train = false;
	int kap1_bigthan_kap0 = 0; double kap1_high = 0;
	int all_in_H = 0; // 0:origin, 1: full, 2: ML
	int tablelike = 0; // 0: point, 1:table, 2:combine, 3:condition
	bool weighted_kappa = false; // Calculate unweighted kappa.
	bool all_points = false; // Don't restrict data point to its cluster when sampling its component
	bool cpnt_prior = false;
	double max_clusters = INFINITY;

	I3gmm(){}
	I3gmm(Vector mu0, Matrix psi0, double m,
		double kappa0, double kappa1,
		double alpha, double gamma); // Run i2gmm
	I3gmm(Vector mu0, Matrix sigma0, double m,
		double c1, double c2, double alpha0, double beta0,
		double alpha1, double beta1,
		double alpha, double gamma); // Run i3gmm
	~I3gmm();

	void add_data(vector<Vector> data_set, double* weights = nullptr);
	void add_data_with_init(vector<Vector> data, Matrix labels, bool isTrain=true);
	void prior_llike();
	void adjust_weights(double* weights);
	void cluster_gibbs(int max_sweep, int burnin = 0, int n_sample = 0, const char* logfile = nullptr);
	void copy_solution(vector<DP<Vector>*> &new_clusters, vector<Customer<Vector>*> &new_custs);	// deep copy of clusters and customers
	void write_labels(vector<Customer<Vector>*> &res, const char* file); // TODO: same as in DP.cpp
	void write_solution(const char* fname);
	void print_hyper();
	void rnd(int n_samples); // TODO
	Vector likelihood();
	double reassign();	// reassign table for all customers.
	double reassign2();	// reassign table for all customers.
	pair<double, bool> assign(Table<Vector> &t);
	Matrix gen_labels(vector<Customer<Vector>*> &res);
	void renew_hyper_params();

	void id_start(double id) { max_cluster_id = id; }

protected:
	double alpha, gamma;
	int max_cluster_id = 0;

	StutGlobal3 *H = nullptr;	// Base distribution
	DP<Vector> *init_cluster = nullptr; // Serve as a container for all tables and customers.
	vector<Table<Vector>*> components;
	unordered_set<DP<Vector>*> clusters; // Global clusters.
	unordered_set<DP<Vector>*> known_clusters;
	unordered_map<Customer<Vector>*, pair<DP<Vector>*,double> > supervisors; // For restricted Gibbs sampling
	vector<Customer<Vector>*> train_custs;

private:
	unordered_map<double, pair<DP<Vector>*, unordered_map<double, Table<Vector>*>>> cmap;

	template <class Iter> double total_loglike(Iter begin, Iter end);
	template <class Iter1, class Iter2> double likelihood_ratio(Iter1 cluster_begin, Iter1 cluster_end, Iter2 cust_begin, Iter2 cust_end);
	template <class T> void sample_hyper(T &hyperparam, T &last_hyper, T low, T high, bool update = false);
	void log_open(ofstream &logfile, const char* logfname);
	void log_record(ofstream &logfile, double loglike);
	void log_hyper(ofstream &logfile);
};