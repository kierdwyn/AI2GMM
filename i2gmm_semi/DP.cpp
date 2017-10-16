#include "DP.h"
#include <random>
#include <utility>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "util.h"
#include "Matrix.h"
using namespace std;

template <class T>
void Table<T>::add_customer(Customer<T>& cust){
	if (custs.insert(&cust).second){
		n_custs += cust.weight;
		H->addData(cust.data,cust.weight);
		cust.table = this;
	}
}

template <class T>
void Table<T>::remove_customer(Customer<T>& cust){
	if (custs.erase(&cust)){
		n_custs -= cust.weight;
		H->removeData(cust.data,cust.weight);
		cust.table = nullptr;
	}
}

template <class T>
double Table<T>::loglike(Customer<T>& customer) const{
	return H->likelihood(customer.data) + log(customer.weight);
}


template <class T>
DP<T>::DP(dist_t dist_choice, double alpha)
	: dist_choice(dist_choice), alpha(alpha),
	H(Distribution<T>::generate(dist_choice)){}

template <class T>
DP<T>::DP(dist_t dist_choice, Distribution<T> *H, double alpha)
	:dist_choice(dist_choice), H(H), alpha(alpha){}

template <class T>
DP<T>::~DP(){
	if (logLikeHistory != nullptr)
		free(logLikeHistory);
	if (H != nullptr)
		delete H;
	for (Customer<T> *cust : customers)
		if (cust->has_table() && cust->table->parent == this) delete cust;
	customers.clear();
	for (Table<T> *t : tables)
		if (t->parent == this) delete t;
	tables.clear();
}

template <class T>
void DP<T>::add_data(vector<T> data_set){
	customers.reserve(data_set.size());
	ordered_custs.reserve(data_set.size());
	for (T data : data_set){
		auto nc = new Customer<T>(data);
		customers.insert(nc);
		ordered_custs.push_back(nc);
		unassigned_custs.push_back(nc);
	}
	if (data_set.size() > 0){
		has_unassigned_cust = true;
	}
}

// Only tables will be changed.
// data_train won't be added to customers but added to ordered_custs.
template <class T>
void DP<T>::add_prior(vector<T> data_train, Vector labels){
	map<double, Table<T>*> tmap;
	for (unsigned i = 0; i < data_train.size(); i++){
		Customer<T> *nc = new Customer<T>(data_train[i]);
		if (tmap.count(labels[i]) == 0){
			tmap[labels[i]] = new Table<T>(Distribution<T>::generate(dist_choice, H), this);
			tmap[labels[i]]->id = (int)labels[i];
			max_table_id = (int)labels[i] > max_table_id ? (int)labels[i] : max_table_id;
			tables.insert(tmap[labels[i]]);
		}
		tmap[labels[i]]->add_customer(*nc);
		ordered_custs.push_back(nc);
	}
}

template <class T>
void DP<T>::add_table(Table<T> &t, bool as_prior){
	tables.insert(&t);
	t.parent = this;
	t.id = ++max_table_id;
	if (!as_prior)
		for (Customer<T> *cust : t.custs)
			customers.insert(cust);
}

template <class T>
void DP<T>::add_customer(Customer<T> &cust){
	customers.insert(&cust);
	cust.leave_table();
	unassigned_custs.push_back(&cust);
	has_unassigned_cust = true;
}

template<class T>
void DP<T>::add_customers(const vector<Customer<T>*>& custs)
{
	customers.insert(custs.begin(), custs.end());
	unassigned_custs.insert(unassigned_custs.end(), custs.begin(), custs.end());
	has_unassigned_cust = true;
}

template <class T>
void DP<T>::remove_table_custs(Table<T> &t){
	for (Customer<T> *cust : t.custs){
		customers.erase(cust);
	}
	t.parent = nullptr;
	if (t.id == max_table_id) max_table_id--;
	tables.erase(&t);
}

template <class T>
void DP<T>::remove_customer(Customer<T> &cust){
	if (customers.erase(&cust)){
		Table<T> *t = cust.table;
		cust.leave_table();
		if (t != nullptr && t->n_custs < 1e-10){
			tables.erase(t);
			if (max_table_id == t->id) max_table_id--;
			delete t;
		}
	}
	else{
		PERROR("in DP::remove_customer: Customer not exist.");
	}
}

template <class T>
void DP<T>::cluster_gibbs(int max_sweep, int burnin = 0, int n_sample = 0, const char* logfname = nullptr){
	ofstream logfile;
	int sample_interval;
	
	// Initialize
	if (logLikeHistory != nullptr)
		free(logLikeHistory);
	logLikeHistory = (double *)malloc(max_sweep*sizeof(double));
	maxLogLike = -INFINITY;
	sample_interval = n_sample>0 ? (max_sweep - burnin - 1) / n_sample + 1 : max_sweep + 1;
	sample_labels.resize(n_sample, ordered_custs.size());
	log_open(logfile, logfname);

	// Begin sampling
	for (int i = 0; i < max_sweep; i++){
		logLikeHistory[i] = reassign().first;
		if (maxLogLike < logLikeHistory[i]){
			copy_solution(best_tables, best_customers);
			maxLogLike = logLikeHistory[i];
			best_sweep = i;
		}
		log_record(logfile, logLikeHistory[i]);
		if (i >= burnin && ((i - burnin) % sample_interval == 0)){
			Matrix ret = gen_labels(ordered_custs);
			sample_labels[(i - burnin) / sample_interval] = ret[0];
		}
	}

	if (logfile.is_open()) logfile.close();
}

template <class T>
pair<double, bool> DP<T>::assgin(Customer<T> &cust){
	double prob;
	bool created_table = false;
	vector<double> probs;
	vector<Table<T>*> tPtrs;
	probs.reserve(tables.size() + 1);
	probs.push_back(H->likelihood(cust.data) + log(cust.weight) + log(alpha));
	double max = probs[0];
	for (Table<T> *tPtr : tables){
		probs.push_back(tPtr->loglike(cust) + log(tPtr->n_custs));
		//probs.push_back(tPtr->loglike(cust));
		if (probs[probs.size() - 1] > max) max = probs[probs.size() - 1];
		tPtrs.push_back(tPtr);
	}

	size_t i(0);
	discrete_distribution<int> dist(probs.size(), 0.0, 1.0, [&probs, &i, &max](double){ return exp(probs[i++] - max); });
	int index = dist(generator);
	if (index > 0){ //assign to table[index-1]
		prob = probs[index] - log(tPtrs[index - 1]->n_custs);
		//prob = probs[index];
		(tPtrs[index - 1])->add_customer(cust);
	}
	else{ // create new table
		prob = probs[index];
		Table<T> *nt = new Table<T>(Distribution<T>::generate(dist_choice, H), this);
		nt->id = ++max_table_id;
		nt->add_customer(cust);
		tables.insert(nt);
		created_table = true;
	}
	return pair<double, bool>( prob, created_table);
}

template <class T>
pair<double, bool> DP<T>::init_assign(){
	pair<double, bool> ret(0, false);
	if (has_unassigned_cust){
		for (Customer<T> *cust : unassigned_custs){
			if (!cust->has_table()){
				pair<double, bool> p = assgin(*cust);
				ret.first += p.first;
				ret.second = p.second || ret.second;
				cust->table->parent->customers.insert(cust);
			}
		}
		unassigned_custs.clear();
		has_unassigned_cust = false;
	}
	return ret;
}

template <class T>
pair<double, bool> DP<T>::reassign(){
	pair<double, bool> ret(0, false);
	for (Customer<T> *cust : customers){
		if (cust->has_table()){
			Table<T> *t = cust->table;
			t->remove_customer(*cust);
			if (t->n_custs < 1e-10){
				tables.erase(t);
				if (max_table_id == t->id) max_table_id--;
				delete t;
			}
		}
		pair<double, bool> p = assgin(*cust);
		ret.first += p.first;
		ret.second = p.second || ret.second;
	}
	unassigned_custs.clear();
	has_unassigned_cust = false;
	return ret;
}

template <class T>
void DP<T>::copy_solution(vector<Table<T>*> &new_tables, vector<Customer<T>*> &new_custs){
	// Clear old solution.
	for (Table<T> *t : new_tables)
		delete t;
	new_tables.clear();
	for (Customer<T> *c : new_custs)
		delete c;
	new_custs.clear();

	// Construct new solution.
	map<Customer<T>*, Customer<T>*> custmap;
	for (Table<T> *t : tables){
		Table<T> *nt = new Table<T>(Distribution<T>::generate(dist_choice, t->H), t->parent);
		nt->id = t->id;
		nt->n_custs = t->n_custs;
		nt->custs.reserve(t->custs.size());
		for (Customer<T> *cust : t->custs){
			Customer<T> *nc = new Customer<T>(cust->data);
			custmap[cust] = nc;
			nt->custs.insert(nc);
			nc->table = nt;
		}
		new_tables.push_back(nt);
	}
	for (auto c : ordered_custs)
		new_custs.push_back(custmap[c]);
}

template <class T>
void DP<T>::write_labels(vector<Customer<T>*> &res, const char* file){
	Matrix labels = gen_labels(res);
	labels.writeMatrix(file);
}

template <class T>
void DP<T>::write_solution(const char* fname){
	sample_labels.writeMatrix((string(fname) + "_samplelabels.txt").c_str());
	gen_labels(ordered_custs).writeMatrix((string(fname) + "_lastlabels.txt").c_str());
	gen_labels(best_customers).writeMatrix((string(fname) + "_bestlabels.txt").c_str());
}

template <class T>
Matrix DP<T>::gen_labels(vector<Customer<T>*> &res){
	int i = 0;
	Matrix labels(1, res.size());
	for (Customer<T>* c : res){
		labels[0][i] = c->table->id;
		i++;
	}
	return labels;
}

template<class T>
void DP<T>::log_open(ofstream & logfile, const char * logfname)
{
	if (logfname != nullptr) {
		logfile.open(logfname);
		if (!logfile.is_open()) {
			string s = "Unable to open file ";
			s += logfname;
			PERROR(s.c_str());
		}
	}
}

template<class T>
void DP<T>::log_record(ofstream & logfile, double loglike)
{
	if (logfile.is_open()) {
		logfile << fixed << setprecision(2) << loglike << " " << tables.size();
		for (auto c : tables)
			logfile << " " << c->custs.size();
		logfile << endl;
	}
}

template class Customer < Vector > ;
template class Table < Vector > ;
template class DP < Vector > ;