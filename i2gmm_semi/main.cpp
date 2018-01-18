#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Matrix.h"
#include "i3gmm.h"

using namespace std;

void test_chol_update() {
	int d = 2, n = 10;
	Matrix M(d); M.zero();
	Matrix L1(d); L1.zero();
	Matrix L2(d); L2.zero();

	for (int i = 0; i < n; i++) {
		Vector x = urand(d);
		M += (x >> x);
	}
	L1 = M.chol();
	for (int i = 0; i < 1; i++) {
		Vector x = urand(d);
		M += (x >> x);
		L2 = M.chol();
		L1 = L1.chol(x);
	}

	cout << "M:" << endl;
	M.print();
	cout << "L1:" << endl;
	L1.print();
	cout << "L2:" << endl;
	L2.print();

	M.zero();
	Vector x1(2), x2(2), x3(2);
	x1.data[0] = 0; x1.data[1] = 1;
	x2.data[0] = 1; x2.data[1] = 0;
	x3.data[0] = 2.5; x3.data[1] = 3.5;
	//x3 = urand(d);
	M += (x1 >> x1);
	M += (x2 >> x2);
	L2 = M.chol();
	cout << "L2" << endl;
	L2.print();
	M += (x3 >> x3);
	L1 = M.chol();
	L2 = L2.chol(x3);
	cout << "M" << endl;
	M.print();
	cout << "L1" << endl;
	L1.print();
	cout << "L2" << endl;
	L2.print();
}

void test_matrix_lib() {
	int d = 2;
	Vector a(d);
	Vector b(d);
	Vector c(d);

	a[0] = 0.5; a[1] = 0.3;
	b[0] = 0.1; b[1] = 0.9;
	c[0] = 0.2; c[1] = 0.5;

	Matrix A = b >> b;
	A = A + (c >> c);
	//A.triangle = true;
	a = a / A;

	A.print();
	cout << endl;
	a.print();
}

void readBin(string filename, Matrix& m)
{
	ifstream file;
	file.open(filename, std::ifstream::binary);
	if (file.is_open())
		file >> m;
	else
		cout << "File not found: " << filename.c_str() << endl;
	file.close();
}

pair<string,string> split_filename(const string& str)
{
	pair<string, string> ret;
	size_t found;
	cout << "Splitting: " << str << endl;
	found = str.find_last_of("/\\");
	ret.first = str.substr(0, found);
	ret.second = str.substr(found + 1);
	cout << " folder: " << str.substr(0, found) << endl;
	cout << " file: " << str.substr(found + 1) << endl;
	return ret;
}

void conf_i3gmm(I3gmm &i3gmm, Matrix &conf, int conf_start) {
	if (conf.n > conf_start) i3gmm.outlier_ratio = conf[0][conf_start];
	if (conf.n > conf_start+1) i3gmm.prior_type = conf[0][conf_start+1];
	if (conf.n > conf_start+2) i3gmm.tune_on_train = conf[0][conf_start+2];
	if (conf.n > conf_start+3) i3gmm.kap1_bigthan_kap0 = conf[0][conf_start+3];
	if (conf.n > conf_start+4) i3gmm.kap1_high = conf[0][conf_start+4];
	if (conf.n > conf_start+5) i3gmm.all_in_H = conf[0][conf_start+5];
	if (conf.n > conf_start+6) i3gmm.tablelike = conf[0][conf_start+6];
	if (conf.n > conf_start+7) i3gmm.weighted_kappa = conf[0][conf_start+7];
	if (conf.n > conf_start+8) i3gmm.cpnt_prior = conf[0][conf_start+8];
	if (conf.n > conf_start+9) i3gmm.all_points = conf[0][conf_start+9];
}

void run_i3gmm(string resultfile, Matrix &prior, Matrix &conf, Matrix &train, vector<Vector> &train_set, vector<Vector> &test_set,
	int MAX_SWEEP, int burnin, int n_samples, double max_clusters, int init_sweep){
	cout << "Parameters: " << " m " << conf[0][0] << " c1 " << conf[0][1] << " c2 " << conf[0][2]
		<< " alpha0 " << conf[0][3] << " beta0 " << conf[0][4] << " alpha1 " << conf[0][5] << " beta1 " << conf[0][6]
		<< " alpha " << conf[0][7] << " gamma " << conf[0][8] << endl;
	I3gmm i3gmm(prior[0], prior.submat(1, prior.r, 0, prior.m),
		conf[0][0], conf[0][1], conf[0][2], conf[0][3], conf[0][4], conf[0][5], conf[0][6], conf[0][7], conf[0][8]);
	conf_i3gmm(i3gmm, conf, 9);
	i3gmm.max_clusters = max_clusters;
	//i3gmm.adjustable = false;
	if (train_set.size() != 0){
		i3gmm.add_data_with_init(train_set, train.submat(0, train.r, 0, train.m-prior.m));
		i3gmm.cluster_gibbs(init_sweep, 0, 0, (resultfile + "_gibbslog_prior.txt").c_str());
		i3gmm.prior_llike();
	}
	i3gmm.add_data(test_set);
	/*i3gmm.cluster_gibbs(init_sweep, 0, init_sweep/10, (resultfile + "_gibbslog_init.txt").c_str());
	i3gmm.sample_labels.writeMatrix((resultfile + "_samplelabels_init.txt").c_str());
	i3gmm.adjustable = true;*/
	i3gmm.cluster_gibbs(MAX_SWEEP, burnin, n_samples, (resultfile + "_gibbslog.txt").c_str());
	i3gmm.write_solution(resultfile.c_str());
	i3gmm.print_hyper();

	cout << "Max log likelihood: " << i3gmm.maxLogLike << " iter: " << i3gmm.best_sweep << " ncluster " << i3gmm.best_clusters.size() << endl;
	for (unsigned i = 0; i < i3gmm.best_clusters.size(); i++){
		cout << i << " nCpnts " << i3gmm.best_clusters[i]->tables.size() << " nCusts " << i3gmm.best_clusters[i]->customers.size() << "\tper Cpnt ";
		for (auto t : i3gmm.best_clusters[i]->tables)
			cout << t->n_custs << ' ';
		cout << endl;
	}
}

void run_i2gmm(string resultfile, Matrix &prior, Matrix &conf, Matrix &train, vector<Vector> &train_set, vector<Vector> &test_set,
	int MAX_SWEEP, int burnin, int n_samples, double max_clusters, int init_sweep=10){
	cout << "Parameters: m " << conf[0][0] << " kappa0 " << conf[0][1] << " kappa1 " << conf[0][2]
		<< " alpha " << conf[0][3] << " gamma " << conf[0][4] << endl;
	I3gmm i3gmm(prior[0], prior.submat(1, prior.r, 0, prior.m),
		conf[0][0], conf[0][1], conf[0][2], conf[0][3], conf[0][4]);
	conf_i3gmm(i3gmm, conf, 5);
	i3gmm.max_clusters = max_clusters;
	if (train_set.size() != 0){
		i3gmm.add_data_with_init(train_set, train.submat(0, train.r, 0, train.m - prior.m));
		i3gmm.cluster_gibbs(init_sweep, 0, 0, (resultfile + "_gibbslog_prior.txt").c_str());
		i3gmm.prior_llike();
	}
	i3gmm.add_data(test_set);
	i3gmm.cluster_gibbs(MAX_SWEEP, burnin, n_samples, (resultfile + "_gibbslog.txt").c_str());
	i3gmm.write_solution(resultfile.c_str());

	cout << "Max log likelihood: " << i3gmm.maxLogLike << " iter: " << i3gmm.best_sweep << " ncluster " << i3gmm.best_clusters.size() << endl;
	for (unsigned i = 0; i < i3gmm.best_clusters.size(); i++){
		cout << i << " nCpnts " << i3gmm.best_clusters[i]->tables.size() << " nCusts " << i3gmm.best_clusters[i]->customers.size() << "\tper Cpnt ";
		for (auto t : i3gmm.best_clusters[i]->tables)
			cout << t->n_custs << ' ';
		cout << endl;
	}
}

void run_dp(string resultfile, Matrix &prior, Matrix &conf, Matrix &train, vector<Vector> &train_set, vector<Vector> &test_set,
	int MAX_SWEEP, int burnin, int n_samples, double max_clusters){
	cout << "Parameters: m " << conf[0][0] << " kappa0 " << conf[0][1] << " alpha " << conf[0][2] << endl;
	HyperParams::init(prior[0], prior.submat(1, prior.r, 0, prior.m), conf[0][0], conf[0][1], 0);
	StutNIW *dist = new StutNIW();
	DP<Vector> dpm(Distribution<Vector>::STU_NIW, dist, conf[0][2]);
	dpm.max_clusters = max_clusters;
	if (train_set.size() != 0){
		dpm.add_prior(train_set, train.submat(0, train.r, 0, 1));
	}
	dpm.add_data(test_set);
	if (conf.n > 3 && conf[0][3]==1) {
		dist = dynamic_cast<StutNIW*>(dpm.get_dist());
		dist->reset();
		dist->add_all(dpm.ordered_custs.begin(), dpm.ordered_custs.end());
	}
	dpm.cluster_gibbs(MAX_SWEEP, burnin, n_samples, (resultfile + "_gibbslog.txt").c_str());
	dpm.write_solution(resultfile.c_str());

	cout << "Max log likelihood: " << dpm.maxLogLike << " iter: " << dpm.best_sweep << endl;
	for (unsigned i = 0; i < dpm.best_tables.size(); i++){
		cout << "Number of customers in cluster " << i << ' ' << dpm.best_tables[i]->n_custs << endl;
	}
}

void run_ixgmm_with_init(string resultfile, Matrix &prior, Matrix &conf, Matrix &train_label, Matrix &test_label, vector<Vector> &train_set, vector<Vector> &test_set, double alpha, double gamma, int n_sweep, int burnin, int n_sample) {
	I3gmm i3gmm;
	cout << "Parameters:" << endl;
	conf.print();
	cout << "Prior:" << endl;
	prior.print();
	int conf_start;
	if (conf.n == 5 || conf.n == 13) {
		// i2gmm
		new (&i3gmm) I3gmm(prior[0], prior.submat(1, prior.r, 0, prior.m),
			conf[0][0], conf[0][1], conf[0][2], conf[0][3], conf[0][4]);
		conf_start = 5;
	}else if (conf.n == 9 || conf.n == 17) {
		// i3gmm
		/*i3gmm = new I3gmm(prior[prior.m - 1], prior.submat(0, prior.r - 2, 0, prior.m),
			conf[0][0], prior[prior.r - 1][0], prior[prior.r - 1][1], alpha, gamma);*/
		new (&i3gmm) I3gmm(prior[0], prior.submat(1, prior.r, 0, prior.m),
			conf[0][0], conf[0][1], conf[0][2], conf[0][3], conf[0][4], conf[0][5], conf[0][6], conf[0][7], conf[0][8]);
		conf_start = 9;
	}else {
		PERROR(("# in conf " + to_string(conf.n) + " not right.\n"
			+ "Should be 5, 9, 13 or 17.\n").c_str());
		return;
	}
	if (conf.n == 13 || conf.n == 17) {
		conf_i3gmm(i3gmm, conf, conf_start);
	}
	i3gmm.add_data_with_init(train_set, train_label);
	if (test_label.r > 0) {
		i3gmm.add_data_with_init(test_set, test_label, false);
	} else {
		if(i3gmm.adjustable)
			i3gmm.renew_hyper_params();
		i3gmm.add_data(test_set);
	}
	i3gmm.cluster_gibbs(n_sweep,burnin,n_sample, (resultfile + "_continued_gibbslog.txt").c_str());
	i3gmm.write_solution((resultfile + "_continued").c_str());
}

/// <param name = "argv">
///		argv[1]: path to the data file.
///		argv[2]: path to the prior file.
///		argv[3]: path to the configuration file.
///		argv[4]: (optional) path to save the results. Default "../data/"
///		argv[5]: (optional) 1: use igmm; 2: use i2gmm; 3: use i3gmm.
///		argv[6]: (optional) # of sweeps. Default 100.
///		argv[7]: (optional) # of burn in. Default 50.
///		argv[8]: (optional) # of samples. Default 5.
///		argv[9]: (optional) path to the training data.
///		argv[10]: (optional) # of init sweep. Default 50.
/// </param>
int main(int argc, char* argv[]) {
	/*debugMode(1);
	init_buffer(1, 2);
	test_chol_update();
	test_matrix_lib();
	pause();
	return 0;*/

	if (!strcmp(argv[1], "continue")) {
		if (argc != 8) {
			PERROR(("# of arguments " + to_string(argc) + " not right.\n" +
				"Usage: i2gmm_semi.exe predict [path to model] [alpha] [gamma] [# of sweeps] [# burnin] [# samples]\n").c_str());
			exit(1);
		}
		pair<string, string> results_prefix = split_filename(argv[2]);
		double alpha = stof(argv[3]);
		double gamma = stof(argv[4]);
		int n_sweep = atoi(argv[5]);
		int burnin = atoi(argv[6]);
		int n_sample = atoi(argv[7]);
		
		string path = results_prefix.first;
		string fname = results_prefix.second;
		Matrix data, train, prior, conf, labels, train_label, test_label;
		vector<Vector> train_set, test_set;

		readBin(results_prefix.first + "\\data.matrix", data);
		readBin(results_prefix.first + "\\data_train.matrix", train);
		readBin(results_prefix.first + "\\data_prior.matrix", prior);
		readBin(results_prefix.first + "\\data_params.matrix", conf);
		labels.readMatrix((string(argv[2]) + "_lastlabels.txt").c_str());

		init_buffer(1, data.m);

		labels = labels.transpose_xy();
		train_label = labels.submat(0, train.r, 0, labels.m);
		test_label = labels.submat(train.r, labels.r, 0, labels.m);


		for (int i = 0; i < data.r; i++) {
			test_set.emplace_back(data[i]);
		}
		Matrix train_data = train.submat(0, train.r, train.m - data.m, train.m);
		for (int i = 0; i < train_data.r; i++)
			train_set.emplace_back(train_data[i]);

		run_ixgmm_with_init(argv[2], prior, conf, train_label, test_label, train_set, test_set, alpha, gamma, n_sweep, burnin, n_sample);
	}
	else {
		string datafile(argv[1]);
		string priorfile = argv[2];
		string configfile = argv[3];
		string traindatafile; // first column in traindatafile represent labels.
		string result_dir = "../data/";
		int model = 1;
		int MAX_SWEEP = 100;
		int INIT_SWEEP = 50;
		int BURNIN = 50;
		int SAMPLE = (MAX_SWEEP - BURNIN) / 10; // Default value is 10 sample + 1 post burnin
		double MAX_CLUSTERS = INFINITY;

		//srand(time(NULL));
		if (argc > 4)
			result_dir = argv[4];
		if (argc > 5)
			model = atoi(argv[5]);
		if (argc > 6)
			MAX_SWEEP = atoi(argv[6]);
		if (argc > 7)
			BURNIN = atoi(argv[7]);
		SAMPLE = (MAX_SWEEP - BURNIN) / 10; // Default value
		if (argc > 8)
			SAMPLE = atoi(argv[8]);
		if (argc > 9)
			traindatafile = argv[9];
		if (argc > 10)
			INIT_SWEEP = atoi(argv[10]);
		if (argc > 11)
			MAX_CLUSTERS = atoi(argv[11]);

		//string datafile;
		//string priorfile;
		//string configfile;
		//string traindatafile; // first column in traindatafile represent labels.
		//string result_dir;
		//int model = 1;
		//int MAX_SWEEP = 10;
		//int INIT_SWEEP = 10;
		//int BURNIN = 0;
		//int SAMPLE = (MAX_SWEEP - BURNIN) / 10; // Default value is 10 sample + 1 post burnin
		//datafile = "C:/Users/chengyic/Google Drive/Studies/Research/i2gmm/data/synthetic/1_d2_0.05_1_4_10_i3gmm5000000_335/data.matrix";
		//priorfile = "C:/Users/chengyic/Google Drive/Studies/Research/i2gmm/data/synthetic/1_d2_0.05_1_4_10_i3gmm5000000_335/data_prior.matrix";
		//configfile = "C:/Users/chengyic/Google Drive/Studies/Research/i2gmm/data/synthetic/1_d2_0.05_1_4_10_i3gmm5000000_335/data_params.matrix";
		//result_dir = "C:/Users/chengyic/Google Drive/Studies/Research/i2gmm/data/synthetic/1_d2_0.05_1_4_10_i3gmm5000000_335/result1";
		//model = 3;
		//MAX_SWEEP = 10;
		//BURNIN = 0;
		//SAMPLE = 5;
		//traindatafile = "C:/Users/chengyic/Google Drive/Studies/Research/i2gmm/data/synthetic/1_d2_0.05_1_4_10_i3gmm5000000_335/data_train.matrix";
		//INIT_SWEEP = 10;

		Matrix data, prior, conf, train;
		vector<Vector> test_set, train_set;
		readBin(datafile, data);
		readBin(priorfile, prior);
		readBin(configfile, conf);

		init_buffer(1, data.m);

		for (int i = 0; i < data.r; i++) {
			test_set.emplace_back(data[i]);
		}
		if (!traindatafile.empty()) {
			readBin(traindatafile, train);
			Matrix train_data = train.submat(0, train.r, train.m - data.m, train.m);
			for (int i = 0; i < train_data.r; i++)
				train_set.emplace_back(train_data[i]);
		}

		step();
		if (model == 1)
			run_dp(result_dir, prior, conf, train, train_set, test_set, MAX_SWEEP, BURNIN, SAMPLE, MAX_CLUSTERS);
		if (model == 2)
			run_i2gmm(result_dir, prior, conf, train, train_set, test_set, MAX_SWEEP, BURNIN, SAMPLE, MAX_CLUSTERS, INIT_SWEEP);
		if (model == 3)
			run_i3gmm(result_dir, prior, conf, train, train_set, test_set, MAX_SWEEP, BURNIN, SAMPLE, MAX_CLUSTERS, INIT_SWEEP);
		step();
	}

	//pause();
	return 0;
}