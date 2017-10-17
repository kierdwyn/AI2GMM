#include "I3gmm.h"
#include <algorithm>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

using namespace std;


I3gmm::I3gmm(Vector mu0, Matrix psi0, double m,
	double kappa0, double kappa1,
	double alpha, double gamma) :alpha(alpha), gamma(gamma){
	H = new StutGlobal3(mu0, psi0, m, kappa0, kappa1);
	init_cluster = new DP<Vector>(Distribution<Vector>::STU_LOCAL,
		Distribution<Vector>::generate(Distribution<Vector>::STU_GLOBAL, H), alpha);
}

I3gmm::I3gmm(Vector mu0, Matrix sigma0, double m,
	double c1, double c2, double alpha0, double beta0,
	double alpha1, double beta1,
	double alpha, double gamma) :alpha(alpha), gamma(gamma){
	H = new StutGlobal3(mu0, sigma0, m, c1, c2,
		alpha0, beta0, alpha1, beta1);
	init_cluster = new DP<Vector>(Distribution<Vector>::STU_LOCAL,
		Distribution<Vector>::generate(Distribution<Vector>::STU_GLOBAL, H), alpha);
	adjustable = true;
}

// Be careful, public members won't be destruct.
I3gmm::~I3gmm(){
	if (logLikeHistory != nullptr){
		free(logLikeHistory);
	}
	delete H;
	delete init_cluster;
	for (DP<Vector> *dp : clusters)
		delete dp;
}

void I3gmm::add_data(vector<Vector> data_set, double* weights){
	if (H == nullptr) {
		PERROR("Need to initialize I3gmm first before add data.\n");
		return;
	}
	vector<Customer<Vector>*> nc;
	for (Vector data : data_set){
		nc.push_back(new Customer<Vector>(data));
	}
	if (weights != nullptr) {
		for (unsigned i = 0; i < data_set.size(); i++) {
			nc[i]->weight = *(weights + i);
		}
	}
	if (!all_points) {
		StutGlobal3 *dist = dynamic_cast<StutGlobal3*>(init_cluster->get_dist());
		dist->update_stut();
		init_cluster->add_customers(nc);
		init_cluster->tables.insert(components.begin(), components.end());
	}
	allcusts.insert(allcusts.end(), nc.begin(), nc.end());
	HyperParams::update_global_stats(allcusts.begin(), allcusts.end());
}

void I3gmm::add_data_with_init(vector<Vector> data, Matrix labels, bool isTrain)
{
	if (!(labels.m == 1 || labels.m == 2)) {
		PERROR(("label " + to_string(labels.r) + "-by-" + to_string(labels.m) + " must be a n-by-1 or 2-by-n matrix.").c_str());
	}
	for (unsigned i = 0; i < data.size(); i++) {
		Customer<Vector> *nc = new Customer<Vector>(data[i]);
		DP<Vector> *nc_cluster;
		Table<Vector> *nc_table;
		allcusts.push_back(nc);
		// Reconstruct clusters
		if (cmap.count(labels[i][0]) == 0) {
			cmap[labels[i][0]].first = new DP<Vector>(Distribution<Vector>::STU_LOCAL, Distribution<Vector>::generate(Distribution<Vector>::STU_GLOBAL, H), alpha);
			nc_cluster = cmap[labels[i][0]].first;
			nc_cluster->id = (int)labels[i][0];
			max_cluster_id = (int)labels[i][0] > max_cluster_id ? (int)labels[i][0] : max_cluster_id;
			clusters.insert(nc_cluster);
			if (isTrain) known_clusters.insert(nc_cluster);
		}
		// Reconstruct components
		if (labels.m == 2 && cmap[labels[i][0]].second.count(labels[i][1]) == 0) {
			// Create new table
			nc_table = new Table<Vector>(Distribution<Vector>::generate(Distribution<Vector>::STU_LOCAL, H), nullptr);
			cmap[labels[i][0]].second[labels[i][1]] = nc_table;
		}
		// Add customer to its cluster or components
		if (labels.m == 1) cmap[labels.data[i]].first->add_customer(*nc);
		else cmap[labels[i][0]].second[labels[i][1]]->add_customer(*nc);
		if (isTrain) {
			supervisors[nc] = pair<DP<Vector>*, double>(cmap[labels[i][0]].first, 0);
			train_custs.push_back(nc);
		}
	}
	if (labels.m == 2) {
		// Add components to their clusters and rebuild the sufficient statistics
		for (auto ele : cmap) {
			for (auto ele1 : ele.second.second) {
				dynamic_cast<StutLocal3*>(ele1.second->H)->change_global(dynamic_cast<StutGlobal3*>(ele.second.first->get_dist()));
				ele.second.first->add_table(*ele1.second,cpnt_prior);
				dynamic_cast<StutGlobal3*>(ele.second.first->get_dist())->add_component(dynamic_cast<StutLocal3*> (ele1.second->H));
			}
			components.insert(components.end(), ele.second.first->tables.begin(), ele.second.first->tables.end());
		}
	}
	HyperParams::update_global_stats(allcusts.begin(), allcusts.end());
}

// Makes sense when only contains training data.
void I3gmm::prior_llike(){
	/*for (DP<Vector>* cluster : known_clusters)
		cluster->maxLogLike = 0;
	for (auto cust : train_custs) {
		supervisors[cust].second = cust->table->parent->get_dist()->likelihood(cust->data);
		supervisors[cust].first->maxLogLike += supervisors[cust].second;
	}
	for (DP<Vector>* cluster : known_clusters) {
		cluster->maxLogLike /= cluster->customers.size();
	}
	
	unordered_map<DP<Vector>*, vector<double>> mahalanobis;
	for (auto cust : train_custs) {
	StutGlobal3 *dist = dynamic_cast<StutGlobal3*>(cust->table->parent->get_dist());
	Vector diff = cust->data - dist->mu_s;
	supervisors[cust].second = - (diff / dist->sigma_s*diff);
	mahalanobis[cust->table->parent].push_back(supervisors[cust].second);
	}
	for (auto c : known_clusters) {
	sort(mahalanobis[c].begin(), mahalanobis[c].end());
	c->maxLogLike = outlier_ratio > 0 ? -outlier_ratio : -INFINITY;
	}*/

	unordered_map<DP<Vector>*, vector<double>> llikes;
	for (auto cust : train_custs) {
		supervisors[cust].second = cust->table->parent->get_dist()->likelihood(cust->data);
		llikes[cust->table->parent].push_back(supervisors[cust].second);
	}
	for (auto c : llikes) {
		int pos = outlier_ratio > 0 ? (c.second.size() / outlier_ratio) : 0;
		nth_element(c.second.begin(), c.second.begin() + pos, c.second.end());
		c.first->maxLogLike = c.second[pos];
	}

	/*unordered_map<DP<Vector>*, vector<double>> llikes;
	for (auto cust : train_custs) {
		supervisors[cust].second = cust->table->parent->get_dist()->likelihood(cust->data);
		llikes[cust->table->parent].push_back(supervisors[cust].second);
	}
	for (auto c : llikes) {
		double max_llike = *max_element(c.second.begin(), c.second.end());
		c.first->maxLogLike = max_llike;
		for (int i = 0; i < c.second.size(); i++) {
			c.second[i] = exp((c.second[i] - max_llike)/d);
		}
		sort(c.second.begin(), c.second.end());
	}
	for (auto cust : train_custs) {
		supervisors[cust].second -= supervisors[cust].first->maxLogLike;
		supervisors[cust].second = exp(supervisors[cust].second / d);
	}*/
}

void I3gmm::adjust_weights(double* weights){
	// Update customers
	int i = 0;
	for (Customer<Vector>* cust : allcusts){
		cust->weight = *(weights + i);
		i++;
	}
	// Update components
	for (Table<Vector>* t : components){
		StutLocal3 *H = dynamic_cast<StutLocal3*>(t->H);
		H->reset();
		H->add_all(t->custs.begin(), (t->custs).end());
		t->n_custs = H->n_points;
	}
	// Update clusters
	for (DP<Vector>* c : clusters){
		StutGlobal3 *H = dynamic_cast<StutGlobal3*>(c->get_dist());
		H->reset();
		H->add_all(c->tables.begin(), c->tables.end());
	}
}

void I3gmm::cluster_gibbs(int max_sweep, int burnin, int n_sample, const char* logfname){
	ofstream logfile;
	int sample_interval;

	// Initialize
	if (logLikeHistory != nullptr)
		free(logLikeHistory);
	logLikeHistory = (double *)malloc(max_sweep*sizeof(double));
	maxLogLike = -INFINITY;
	sample_interval = n_sample>0 ? (max_sweep-burnin-1)/n_sample+1 : max_sweep+1;
	sample_labels.resize(2 * n_sample, allcusts.size());
	log_open(logfile, logfname);

	// Begin sampling
	for (int i = 0; i < max_sweep; i++){
		if (all_points) reassign2();
		else reassign();
		logLikeHistory[i] = likelihood().mean();
		// Renew hyper parameters
		if (adjustable) renew_hyper_params();
		if (maxLogLike < logLikeHistory[i]){
			copy_solution(best_clusters, best_customers);
			if (i != 0) {
				maxLogLike = logLikeHistory[i];
				best_sweep = i;
			}
		}
		log_record(logfile, logLikeHistory[i]);
		if (n_sample>0 && i >= burnin && ((i - burnin) % sample_interval == 0)){
			Matrix ret = gen_labels(allcusts);
			sample_labels[(i - burnin) / sample_interval * 2] = ret[0];
			sample_labels[(i - burnin) / sample_interval * 2 + 1] = ret[1];
		}

		if (clusters.size() > max_clusters) {
			PERROR(("Reached # of clusters upperbound (more than"+ to_string(max_clusters) +"), stop.").c_str());
			exit(1);
		}
	}

	log_hyper(logfile);
	if (logfile.is_open()) logfile.close();
}

Vector I3gmm::likelihood(){
	Vector llike(allcusts.size());
	for (int i = 0; i < allcusts.size(); i++) {
		if (allcusts[i]->table == nullptr) {
			llike[i] = allcusts[i]->parent->get_dist()->likelihood((*allcusts[i]).data);
		}
		else {
			llike[i] = allcusts[i]->table->parent->get_dist()->likelihood((*allcusts[i]).data);
		}
		//llike[i] = allcusts[i]->table->parent->get_dist()->likelihood((*allcusts[i]).data);
		//llike[i] = allcusts[i]->table->H->likelihood(allcusts[i]->data);
		//llike[i] = allcusts[i]->table->loglike(*allcusts[i]);
	}
	return llike;
}

double I3gmm::reassign(){
	double AvgLogLike = 0;
	components.clear();

	// Assign new customers to exist tables.
	pair<double, bool> ret = init_cluster->init_assign();
	if (ret.second) {
		for (Table<Vector> *t : init_cluster->tables) {
			if (t->parent == init_cluster)
				components.push_back(t);
		}
	}
	init_cluster->tables.clear();

	// Sample customers with components.
	// Collect all components from clusters and renew cluster distribution.
	// TODO: more efficient way.
	for (DP<Vector>* cluster : clusters){
		ret = cluster->reassign();
		StutGlobal3 *dist = dynamic_cast<StutGlobal3*>(cluster->get_dist());
		dist->reset();
		dist->add_all(cluster->tables.begin(), cluster->tables.end());
		cluster->n_tables = dist->n_clusters;
		AvgLogLike += ret.first;
		components.insert(components.end(), cluster->tables.begin(), cluster->tables.end());
	}
	if (all_in_H==1) {
		H->reset();
		H->add_all(components.begin(), components.end());
	}

	// Mark components with training data as restricted components.
	unordered_map<Table<Vector>*, vector<Customer<Vector>*>> known_cpnt_custs;
	for (auto kv : supervisors){
		known_cpnt_custs[kv.first->table].push_back(kv.first);
	}
	// Don't count restricted components for Dirichlet prior. 
	/*for (auto kv : known_cpnt_custs) {
		kv.first->parent->n_tables--;
	}
	for (auto kv : known_clusters) {
		kv->n_tables++;
	}*/

	// Sample components with global clusters.
	for (Table<Vector> *t : components){

		// Don't sample restricted components.
		if (known_cpnt_custs.count(t)) {
			double llike = -INFINITY;
			if (train_custs.size() == allcusts.size()
				|| (cpnt_prior && cmap[t->parent->id].second.count(t->id)))
				continue;
			for (auto cust : known_cpnt_custs[t]) {
				llike = max(supervisors[cust].second, llike);
			}
			/*if (llike >= supervisors[known_cpnt_custs[t][0]].first->maxLogLike
				|| ntrain1/nall1 > 0.5*ntrain/nall)*/
			if (llike >= supervisors[known_cpnt_custs[t][0]].first->maxLogLike)
				continue;
		}

		DP<Vector> *oldp = t->parent;
		if (oldp != nullptr && oldp != init_cluster){
			dynamic_cast<StutGlobal3*>(oldp->get_dist())->remove_component(dynamic_cast<StutLocal3*> (t->H));
			//if (!known_cpnt_custs.count(t)) {
				oldp->n_tables--;
			//}
		}
		assign(*t);
		DP<Vector> *np = t->parent;
		if (oldp != t->parent){
			if (oldp != nullptr){
				oldp->remove_table_custs(*t);
				if (oldp->tables.size() <= 0 && !known_clusters.count(oldp) && oldp != init_cluster){
					clusters.erase(oldp);
					delete oldp;
				}
			}
			np->add_table(*t);
		}
		dynamic_cast<StutLocal3*>(t->H)->change_global(dynamic_cast<StutGlobal3*>(np->get_dist()));
		dynamic_cast<StutGlobal3*>(np->get_dist())->add_component(dynamic_cast<StutLocal3*> (t->H));
		//if ((!known_cpnt_custs.count(t)) || t->parent != supervisors[(known_cpnt_custs[t])[0]].first) {
			np->n_tables++;
		//}
	}

	// Move training data points to its true class
	for (auto s : supervisors) {
		if (s.first->table->parent != s.second.first) {
			s.first->table->parent->remove_customer(*s.first);
			s.first->parent = s.second.first;
			s.second.first->add_customer(*s.first);
		}
	}
	// Delete the artificial clusters contains only training data points
	for (auto citer = clusters.begin(); citer != clusters.end();) {
		if ((*citer)->customers.size() <= 0 && !known_clusters.count(*citer)) {
			delete (*citer);
			citer = clusters.erase(citer);
		}
		else {
			++citer;
		}
	}
	//// Assign table for moved training data
	//for (auto citer = known_clusters.begin(); citer != known_clusters.end(); citer++) {
	//	if ((*citer)->has_unassigned_cust) {
	//		(*citer)->init_assign();
	//	}
	//}

	AvgLogLike /= allcusts.size();
	return AvgLogLike;
}


double I3gmm::reassign2(){
	double AvgLogLike = 0;
	components.clear();

	// Sample customers with components.
	// Collect all components from clusters and renew cluster distribution.
	for (Customer<Vector>* cust : allcusts) {
		if (cpnt_prior && supervisors.count(cust))
			continue;
		vector<double> probs;
		vector<Table<Vector>*> tPtrs;
		vector<DP<Vector>*> cPtrs;
		double max = -INFINITY;
		probs.resize(clusters.size() + 1, -INFINITY);

		// Remove cust from its table
		if (cust->has_table()) {
			cust->table->parent->remove_customer(*cust);
		}

		if (supervisors.count(cust)) {
			// For training data sample within its cluster
			DP<Vector>* cluster = supervisors[cust].first;
			probs[cPtrs.size()] = cluster->get_dist()->likelihood(cust->data) + log(cust->weight) + log(alpha);
			max = probs[cPtrs.size()];
			cPtrs.push_back(cluster);
			for (Table<Vector>* tPtr : cluster->tables) {
				probs.push_back(tPtr->loglike(*cust) + log(tPtr->n_custs));
				if (probs[probs.size() - 1] > max) max = probs[probs.size() - 1];
				tPtrs.push_back(tPtr);
			}
		}
		else {
			// Sample for new table among all tables in all clusters
			for (DP<Vector>* cluster : clusters) {
				probs[cPtrs.size()] = cluster->get_dist()->likelihood(cust->data) + log(cust->weight) + log(alpha);
				if (probs[cPtrs.size()] > max) max = probs[cPtrs.size()];
				cPtrs.push_back(cluster);
				for (Table<Vector>* tPtr : cluster->tables) {
					probs.push_back(tPtr->loglike(*cust) + log(tPtr->n_custs));
					if (probs[probs.size() - 1] > max) max = probs[probs.size() - 1];
					tPtrs.push_back(tPtr);
				}
			}
			// For new table
			probs[clusters.size()] = H->likelihood(cust->data) + log(cust->weight) + log(alpha);
			if (probs[clusters.size()] > max) max = probs[clusters.size()];
			cPtrs.push_back(init_cluster);
			// For un-assigned tables
			for (Table<Vector>* tPtr : init_cluster->tables) {
				probs.push_back(tPtr->loglike(*cust) + log(tPtr->n_custs));
				if (probs[probs.size() - 1] > max) max = probs[probs.size() - 1];
				tPtrs.push_back(tPtr);
			}
		}

		size_t i(0);
		discrete_distribution<int> dist(probs.size(), 0.0, 1.0, [&probs, &i, &max](double) { return exp(probs[i++] - max); });
		int index = dist(generator);
		AvgLogLike += probs[index];

		if (index > clusters.size()) {
			tPtrs[index - clusters.size()-1]->add_customer(*cust);
			if (tPtrs[index - clusters.size()-1]->parent != nullptr)
				tPtrs[index - clusters.size()-1]->parent->customers.insert(cust);
		}
		else { // Create new table
			Table<Vector> *nt = new Table<Vector>(Distribution<Vector>::generate(
				Distribution<Vector>::STU_LOCAL, cPtrs[index]->get_dist()), cPtrs[index]);
			nt->add_customer(*cust);
			cPtrs[index]->add_table(*nt);
		}
	}


	for (Table<Vector> *t : init_cluster->tables) {
		if (t->parent == init_cluster)
			components.push_back(t);
	}
	for (DP<Vector>* cluster : clusters) {
		StutGlobal3 *dist = dynamic_cast<StutGlobal3*>(cluster->get_dist());
		dist->reset();
		dist->add_all(cluster->tables.begin(), cluster->tables.end());
		components.insert(components.end(), cluster->tables.begin(), cluster->tables.end());
	}
	if (all_in_H==1) {
		H->reset();
		H->add_all(components.begin(), components.end());
	}

	// Mark components with training data as restricted components.
	unordered_map<Table<Vector>*, vector<Customer<Vector>*>> known_cpnt_custs;
	for (auto kv : supervisors) {
		known_cpnt_custs[kv.first->table].push_back(kv.first);
	}

	// Sample components with global clusters.
	for (Table<Vector> *t : components) {
		// Don't sample restricted components.
		if (known_cpnt_custs.count(t)) {
			double llike = -INFINITY;
			for (auto cust : known_cpnt_custs[t]) {
				llike = max(supervisors[cust].second, llike);
			}
			if (llike >= supervisors[known_cpnt_custs[t][0]].first->maxLogLike
				|| cmap[t->parent->id].second.count(t->id))
				continue;
		}

		DP<Vector> *oldp = t->parent;
		if (oldp != nullptr && oldp != init_cluster) {
			dynamic_cast<StutGlobal3*>(oldp->get_dist())->remove_component(dynamic_cast<StutLocal3*> (t->H));
		}
		assign(*t);
		DP<Vector> *np = t->parent;
		if (oldp != t->parent) {
			if (oldp != nullptr) {
				oldp->remove_table_custs(*t);
				if (oldp->tables.size() <= 0 && !known_clusters.count(oldp) && oldp != init_cluster) {
					clusters.erase(oldp);
					delete oldp;
				}
			}
			np->add_table(*t);
		}
		dynamic_cast<StutLocal3*>(t->H)->change_global(dynamic_cast<StutGlobal3*>(np->get_dist()));
		dynamic_cast<StutGlobal3*>(np->get_dist())->add_component(dynamic_cast<StutLocal3*> (t->H));
	}

	// Move training data points to its true class
	for (auto s : supervisors) {
		if (s.first->table->parent != s.second.first) {
			s.first->table->parent->remove_customer(*s.first);
			s.first->parent = s.second.first;
			s.second.first->add_customer(*s.first);
		}
	}
	// Delete the artificial clusters contains only training data points
	for (auto citer = clusters.begin(); citer != clusters.end();) {
		if ((*citer)->customers.size() <= 0 && !known_clusters.count(*citer)) {
			delete (*citer);
			citer = clusters.erase(citer);
		}
		else {
			++citer;
		}
	}
	//// Assign table for moved training data
	//for (auto citer = known_clusters.begin(); citer != known_clusters.end(); citer++) {
	//	if ((*citer)->has_unassigned_cust) {
	//		(*citer)->init_assign();
	//	}
	//}

	AvgLogLike /= allcusts.size();
	return AvgLogLike;
}

// TODO: further decomposition to remove repeating code with assign in DP.h
pair<double, bool> I3gmm::assign(Table<Vector> &t){
	bool created_table = false;
	vector<double> probs;
	vector<DP<Vector>*> ptrs;
	probs.push_back(H->loglike(t,tablelike) + log(gamma));
	double max = probs[0];
	for (DP<Vector> *ptr : clusters){
		StutGlobal3 *dist = dynamic_cast<StutGlobal3*>(ptr->get_dist());
		if (prior_type == 0)
			probs.push_back(dist->loglike(t, tablelike) + log(ptr->n_tables));
		else if (prior_type == 1)
			probs.push_back(dist->loglike(t, tablelike) + log(log2(ptr->n_tables) + 1));
		else if (prior_type == 2)
			probs.push_back(dist->loglike(t, tablelike));
		else {
			PERROR("Undefined prior type.\n");
			exit(1);
		}
		if (probs[probs.size() - 1] > max) max = probs[probs.size() - 1];
		ptrs.push_back(ptr);
	}

	size_t i(0);
	discrete_distribution<int> dist(probs.size(), 0.0, 1.0, [&probs, &i, &max](double){ return exp(probs[i++] - max); });
	int index = dist(generator);
	if (index == 0){ // create new DP
		DP<Vector> *nc = new DP<Vector>(Distribution<Vector>::STU_LOCAL,
			Distribution<Vector>::generate(Distribution<Vector>::STU_GLOBAL, H), alpha);
		nc->id = ++max_cluster_id;
		ptrs.push_back(nc);
		clusters.insert(nc);
		created_table = true;
	}
	t.parent = ptrs[(index + ptrs.size() - 1) % ptrs.size()];

	return pair<double, bool>(probs[index], created_table);
}

void I3gmm::copy_solution(vector<DP<Vector>*> &new_clusters, vector<Customer<Vector>*> &new_custs){
	// Clear old solution.
	for (DP<Vector> *dp : new_clusters) {
		for (Table<Vector> *t : dp->tables) {
			delete t;
		}
		dp->tables.clear();
		for (Customer<Vector> *cust : dp->customers) {
			delete cust;
		}
		dp->customers.clear();
		delete dp;
	}
	new_clusters.clear(); new_custs.clear();

	// Construct new solution.
	map<Customer<Vector>*, Customer<Vector>*> old2new; // to make new customers are the same order as the old ones.
	for (DP<Vector> *dp : clusters){
		DP<Vector> *ndp = new DP<Vector>(
			Distribution<Vector>::STU_LOCAL,
			Distribution<Vector>::generate(
			Distribution<Vector>::STU_GLOBAL, dp->get_dist()), alpha);
		ndp->id = dp->id;
		ndp->tables.reserve(dp->tables.size());
		ndp->customers.reserve(dp->customers.size());
		for (Table<Vector> *t : dp->tables){
			Table<Vector> *nt = new Table<Vector>(
				Distribution<Vector>::generate(
				Distribution<Vector>::STU_LOCAL, t->H), ndp);
			nt->n_custs = t->n_custs;
			nt->custs.reserve(t->n_custs);
			dynamic_cast<StutLocal3*>(nt->H)->g_dist = dynamic_cast<StutGlobal3*>(ndp->get_dist());
			/*dynamic_cast<StutLocal3*>(nt->H)->change_global(
				dynamic_cast<StutGlobal3*>(ndp->get_dist()));*/
			for (Customer<Vector> *c : t->custs){
				Customer<Vector> *nc = new Customer<Vector>(*c);
				old2new[c] = nc;
				nt->custs.insert(nc);
				nc->table = nt;
			}
			ndp->add_table(*nt);
		}
		for (Customer<Vector>* c : dp->unassigned_custs) {
			Customer<Vector> *nc = new Customer<Vector>(*c);
			old2new[c] = nc; nc->parent = ndp;
			ndp->add_customer(*nc);
		}
		new_clusters.push_back(ndp);
	}

	// Copy customers
	for (Customer<Vector>* c : allcusts){
		new_custs.push_back(old2new[c]);
	}
}

void I3gmm::write_labels(vector<Customer<Vector>*> &res, const char* file){
	Matrix labels = gen_labels(res);
	labels.writeMatrix(file);
}

// Write sample labels to file
void I3gmm::write_solution(const char* fname){
	int d = H->mu0.n;
	sample_labels.writeMatrix((string(fname) + "_samplelabels.txt").c_str());
	gen_labels(allcusts).writeMatrix((string(fname) + "_lastlabels.txt").c_str());
	gen_labels(best_customers).writeMatrix((string(fname) + "_bestlabels.txt").c_str());

	Matrix hyper_params = HyperParams::psi0;
	hyper_params.resize(d + 2, d);
	hyper_params[d] = HyperParams::mu0;
	hyper_params[d + 1].zero();
	hyper_params[d + 1][0] = HyperParams::kappa0;
	hyper_params[d + 1][1] = HyperParams::kappa1;
	hyper_params.writeMatrix((string(fname) + "_hyperparams.txt").c_str());
	likelihood().writeBin((string(fname) + "_likelihoods.matrix").c_str());
}

void I3gmm::print_hyper(){
	cout << "mu0:" << endl;
	H->mu0.print();
	cout << "psi0:" << endl;
	H->psi0.print();
	cout << "kappa0: " << H->kappa0 << " kappa1: " << H->kappa1 << " m: " << H->m << endl;
	cout << "alpha: " << alpha << " gamma: " << gamma << endl;
}

void I3gmm::rnd(int n_samples){
	// TODO
}

Matrix I3gmm::gen_labels(vector<Customer<Vector>*> &res){
	int num_cpnt = 0, i = 0;
	map<Table<Vector>*, int> cpnt_ids;
	Matrix labels(2, res.size());
	for (Customer<Vector>* c : res){
		if (c->table == nullptr) {
			labels[0][i] = c->parent->id;
			labels[1][i] = -1;
		}
		else {
			if (cpnt_ids.count(c->table) == 0)
				cpnt_ids[c->table] = num_cpnt++;
			labels[0][i] = c->table->parent->id;
			labels[1][i] = cpnt_ids[c->table];
		}
		i++;
	}
	return labels;
}

void I3gmm::renew_hyper_params(){
	double old_kap0 = INFINITY, old_psi00 = INFINITY,i = 0;
	double &kap1 = HyperParams::kappa1, &kap0 = HyperParams::kappa0;
	while (abs(old_kap0 - kap0) / kap0 > 0.1
		|| abs(old_psi00 - HyperParams::psi0[0][0]) / HyperParams::psi0[0][0] > 0.1) {
		old_kap0 = kap0; old_psi00 = HyperParams::psi0[0][0];
		if (tune_on_train && known_clusters.size() > 0)
			HyperParams::renew_hyper_params(known_clusters.begin(), known_clusters.end(), all_in_H);
		else
			HyperParams::renew_hyper_params(clusters.begin(), clusters.end(), all_in_H);
		if (kap1_bigthan_kap0 > 0 && kap1 < kap0)
			kap1 = kap1_bigthan_kap0*kap0;
		if (kap1_high > 0 && kap1 > kap1_high*kap0)
			kap1 = kap1_high*kap0;
		//sample_hyper(HyperParams::kappa1, HyperParams::last_kappa1, HyperParams::kappa0, HyperParams::kappa0>3?HyperParams::kappa0:3, true);
		//sample_hyper(HyperParams::kappa1, HyperParams::last_kappa1, HyperParams::kappa0, HyperParams::kappa0*10, true);
		i++;
		if (i > 20) {
			break;
		}
	}

	/*if (tune_on_train && known_clusters.size() > 0)
		HyperParams::renew_hyper_params(known_clusters.begin(), known_clusters.end(), all_in_H);
	else
		HyperParams::renew_hyper_params(clusters.begin(), clusters.end(), all_in_H);
	if (kap1_bigthan_kap0 > 0 && kap1 < kap0)
		kap1 = kap1_bigthan_kap0*kap0;
	if (kap1_high > 0 && kap1 > kap1_high*kap0)
		kap1 = kap1_high*kap0;*/

	//all_in_H = kap1 > 2 * kap0 ? false : true;

	//sample_hyper(HyperParams::m, HyperParams::last_m, d + 2, d * 100);

	if (all_in_H != 1) {
		H->update_stut(all_in_H);
	}
	for (auto c : clusters){
		StutGlobal3 *dist = dynamic_cast<StutGlobal3 *>(c->get_dist());
		dist->reset();
		dist->add_all(c->tables.begin(), c->tables.end());
		for (auto cpn : c->tables) {
			StutLocal3 *distc = dynamic_cast<StutLocal3 *>(cpn->H);
			distc->update_stut();
		}
	}
	StutGlobal3 *dist = dynamic_cast<StutGlobal3 *>(init_cluster->get_dist());
	dist->update_stut();

	/*int d = HyperParams::mu0.n;
	for (int i = 0; i < HyperParams::psi0.r; i++) {
		sample_hyper(HyperParams::psi0[i][i], HyperParams::last_psi0[i][i], 1e-3, d*20.0);
	}
	sample_hyper(HyperParams::kappa0, HyperParams::last_kappa0, 1e-3, 3.0);
	sample_hyper(HyperParams::kappa1, HyperParams::last_kappa1, 1e-3, 3.0, true);
	sample_hyper(HyperParams::m, HyperParams::last_m, d + 2, d * 20);*/
}

template <class Iter>
double I3gmm::total_loglike(Iter begin, Iter end) {
	double ret = 0;
	for (auto citer = begin; citer != end; citer++) {
		DP<Vector> *cluster = dynamic_cast<DP<Vector>*>(*citer);
		StutGlobal3 *cdist = dynamic_cast<StutGlobal3 *>(cluster->get_dist());
		for (auto table : cluster->tables) {
			ret += cdist->loglike(*table, tablelike);
		}
	}
	return ret;
}

template <class Iter1, class Iter2>
double I3gmm::likelihood_ratio(Iter1 cluster_begin, Iter1 cluster_end, Iter2 cust_begin, Iter2 cust_end) {
	double ret = 0;
	double n = 0, n1 = 0;
	for (auto cust_iter = cust_begin; cust_iter != cust_end; cust_iter++, n++) {
		double parent_like = 0, total_like = 0;
		Customer<Vector> *cust = dynamic_cast<Customer<Vector>*>(*cust_iter);
		for (auto cluster_iter = cluster_begin; cluster_iter != cluster_end; cluster_iter++) {
			DP<Vector> *cluster = dynamic_cast<DP<Vector>*>(*cluster_iter);
			double likelihood = exp(cluster->get_dist()->likelihood(cust->data));
			if (cluster == cust->table->parent)
				parent_like = likelihood;
			total_like += likelihood;
		}
		if (total_like != 0) {
			n1++;
			ret += 2 * log(parent_like) - log(total_like);
		}
	}
	return (ret*n / n1);
}

template <class T>
void I3gmm::sample_hyper(T &hyperparam, T &last_hyper, T low, T high, bool renew_dist) {
	const int n_step = 10, n_part = 3;
	vector<T> params;
	vector<double> score;
	double max_score = -INFINITY;

	unordered_set<DP<Vector>*> *cluster_ref = &clusters;
	vector<Customer<Vector>*> *cust_ref = &allcusts;
	/*if (known_clusters.size() > 0) {
		cluster_ref = &known_clusters;
		cust_ref = &train_custs;
	}*/

	// Generate params for sampling
	T xp[n_part + 1];
	xp[0] = low; xp[n_part] = high;
	xp[1] = hyperparam > low && hyperparam < high ? hyperparam : (hyperparam < high ? low : high);
	xp[2] = last_hyper > low && last_hyper < high ? last_hyper : (last_hyper < high ? low : high);
	last_hyper = hyperparam;
	for (int i = 0; i < n_part; i++) {
		T dx = (xp[i + 1] - xp[i]) / n_step;
		if (dx == 0) {
			if (xp[i] != xp[i + 1])
				params.push_back(xp[i]);
		}
		else {
			for (int j = 0; j < n_step; j++) {
				params.push_back(xp[i] + j*dx);
			}
		}
	}

	// Calculate scores
	for (size_t i = 0; i < params.size(); i++) {
		hyperparam = params[i];
		for (auto c : *cluster_ref) {
			StutGlobal3 *dist = dynamic_cast<StutGlobal3 *>(c->get_dist());
			if (renew_dist) {
				dist->reset();
				dist->add_all(c->tables.begin(), c->tables.end());
			}
			else
				dist->update_stut();
		}
		score.push_back(total_loglike(cluster_ref->begin(), cluster_ref->end()));
		//score[i] = likelihood_ratio(cluster_ref->begin(), cluster_ref->end(), cust_ref->begin(), cust_ref->end());
		if (score[i] > max_score)
			max_score = score[i];
	}

	// Generate a sample
	size_t i(0);
	discrete_distribution<int> dist(params.size(), 0.0, 1.0, [&score, &i, &max_score](double) { return exp(score[i++] - max_score); });
	int index = dist(generator);
	hyperparam = params[index];
	for (auto c : clusters) {
		StutGlobal3 *dist = dynamic_cast<StutGlobal3 *>(c->get_dist());
		if (renew_dist) {
			dist->reset();
			dist->add_all(c->tables.begin(), c->tables.end());
		}
		else
			dist->update_stut();
	}
}

void I3gmm::log_open(ofstream & logfile, const char * logfname)
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

void I3gmm::log_record(ofstream & logfile, double loglike)
{
	if (logfile.is_open()) {
		logfile << fixed << setprecision(2) << loglike << " " << fixed << setprecision(2) << HyperParams::kappa0
			<< " " << fixed << setprecision(2) << HyperParams::kappa1 << " " << components.size();
		for (auto c : clusters)
			logfile << " " << c->customers.size();
		logfile << endl;
	}
}

void I3gmm::log_hyper(ofstream & logfile)
{
	// Save hyper parameters to logfile.
	if (logfile.is_open()) {
		streambuf *coutbuf = cout.rdbuf();
		cout.rdbuf(logfile.rdbuf());
		print_hyper();
		cout.rdbuf(coutbuf);
	}
}

//template double I3gmm::total_loglike<set<DP<Vector>*, dpcomp>::iterator>(
//	set<DP<Vector>*, dpcomp>::iterator, set<DP<Vector>*>::iterator);
template double I3gmm::total_loglike<unordered_set<DP<Vector>*>::iterator>(
	unordered_set<DP<Vector>*>::iterator, unordered_set<DP<Vector>*>::iterator);
template void I3gmm::sample_hyper<double>(double &, double &, double, double, bool);
template void I3gmm::sample_hyper<int>(int &, int &, int, int, bool);
