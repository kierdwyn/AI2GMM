#pragma once
#include <vector>
#include <unordered_set>
#include "Distribution.h"
#include "Vector.h"
#include "util.h"

#ifdef _WIN64
#include "mex.h"
#define PERROR(msg) mexErrMsgTxt(msg)
#else
#define PERROR(msg) perror(msg)
#endif // _WIN64

template <class T> class Table;
template <class T> class DP;

template <class T>
class Customer{
public:
	T data;
	Table<T> *table = nullptr;
	DP<T> *parent = nullptr; // Only for label generation of un-assigned supervisors.
	double weight = 1; // The weight of the customer. Default to 1.

	Customer(){};
	Customer(T data, double weight = 1):data(data), weight(weight) {}

	bool has_table(){ return table != nullptr; }
	void join_table(Table<T> &t){ t.add_customer(*this); }
	void leave_table(){ if (table) table->remove_customer(*this); }
};


template <class T>
class Table{
public:
	double n_custs = 0; // Number of customers.
	int id; // Note: unique within an instance of DP.
	DP<T> *parent;
	Distribution<T> *H;
	unordered_set<Customer<T>*> custs;

	Table() :H(nullptr), parent(nullptr){}
	Table(Distribution<T> *H, DP<T> *p) :H(H), parent(p){};
	~Table(){ if (H != nullptr) delete H; }

	void add_customer(Customer<T>& customer);
	void remove_customer(Customer<T>& customer);
	double loglike(Customer<T>& customer) const;
};


/// <summary> The class represents dirichlet process.
/// </summary>
template <class T>
class DP{
public:
	unordered_set<Table<T>*> tables;
	unordered_set<Customer<T>*> customers;
	vector<Table<T>*> best_tables;
	vector<Customer<T>*> best_customers;
	vector<Customer<T>*> ordered_custs; // to preserve the order when running single layer.
	vector<Customer<T>*> unassigned_custs;
	Matrix sample_labels;
	double* logLikeHistory = nullptr;
	double maxLogLike = -INFINITY;
	int best_sweep = 0;
	bool has_unassigned_cust = false; // To make init_assign faster.
	int id;
	int max_table_id = 0;
	double n_tables = 0;
	double alpha;		// Concentration parameter

	DP(dist_t dist_choice, double alpha);
	DP(dist_t dist_choice, Distribution<T> *H, double alpha);
	~DP();

	Distribution<T>* get_dist(){ return H; }
	void add_data(vector<T> data_set); // Build new customers according to data_set without assignment.
	void add_prior(vector<T> data_train, Vector labels); // add supervised infomation.
	void add_table(Table<T> &t, bool as_prior = false); // Add table along with its customers without construct.
	void add_customer(Customer<T> &cust); // Add a customer pointer into customers and set its table to nullptr. Won't construct new customer.
	void add_customers(const vector<Customer<T>*> &custs);
	void remove_table_custs(Table<T> &t); // Remove the table along with its customers from the list without desctruct them.
	void remove_customer(Customer<T> &cust); // Remove a customer from customers without destruct it.
	//void remove_all(){ tables.clear(); customers.clear(); }; // Remove all tables and customers without destruct them. 
	void cluster_gibbs(int max_sweep, int burnin = 0, int n_sample = 0, const char* logfname = nullptr);
	void copy_solution(vector<Table<T>*> &new_tables, vector<Customer<T>*> &new_custs);	// deep copy of tables and customers
	void write_labels(vector<Customer<T>*> &res, const char* file);
	void write_solution(const char* fname);
	pair<double, bool> assgin(Customer<T> &cust);
	pair<double, bool> init_assign(); // Only assign customers with no tables.
	pair<double, bool> reassign();	// Reassign all customers to tables.
	Matrix gen_labels(vector<Customer<T>*> &res);

private:
	dist_t dist_choice;
	Distribution<T> *H;	// Base distribution

	void log_open(ofstream &logfile, const char* logfname);
	void log_record(ofstream &logfile, double loglike);
};
