#include "mex.h"
#include "class_handle.hpp"
#include "i3gmm.h"
#include <vector>

unordered_map<uint64_t, bool> isI3gmm_map;
inline bool get_isI3gmm(const mxArray *a) {
	if (!isI3gmm_map.count(*((uint64_t *)mxGetData(a))))
		mexErrMsgTxt("handle not exist.");
	return isI3gmm_map[*((uint64_t *)mxGetData(a))];
}

void set_isI3gmm(mxArray *a, bool isI3gmm) {
	isI3gmm_map[*((uint64_t *)mxGetData(a))] = isI3gmm;
}

void get_dataset(const mxArray *ds, vector<Vector>& dataset) {
	double *ptr = mxGetPr(ds);
	int d = mxGetM(ds), n = mxGetN(ds);
	for (int i = 0; i < n; i++) {
		dataset.push_back(Vector(ptr + i*d, d));
	}
	return;
}

/// <param name="prhs">
///		prhs[2]: The vector conf contains the configurations for i3gmm or igmm.
///				 The choice of i3gmm or i2gmm or igmm depending on the size of conf.
///				 If 3 elements in conf, igmm is chosen, conf = [m, kappa0, alpha];
///				 If 5 elements in conf, i2gmm is chosen, conf = [m, kappa0, kappa1, alpha, gamma];
///				 If 9 elements in conf, i3gmm is chosen,
///					conf = [m, c1, c2, alpha0, beta0, alpha1, beta1, alpha, gamma].
/// </param>
void new_i3gmm(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	bool isI3gmm = true;

	// Check parameters
	if (nrhs != 3)
		mexErrMsgTxt("New: 3 inputs expected: command, prior, configuration");
	int d = mxGetM(prhs[1]);
	double* prior = mxGetPr(prhs[1]);
	double* conf = mxGetPr(prhs[2]); // m, kappa0, alpha

	// Initialize global environment for I3gmm
	init_buffer(1, d);

	// Return a handle to a new C++ instance
	switch (mxGetNumberOfElements(prhs[2]))
	{
	case 3:
		HyperParams::init(Vector(prior, d), Matrix(prior + d, d), conf[0], conf[1], 0);
		plhs[0] = convertPtr2Mat<DP<Vector>>(new DP<Vector>(
			Distribution<Vector>::STU_NIW, new StutNIW(), conf[2]));
		isI3gmm = false;
		break;
	case 5:
		plhs[0] = convertPtr2Mat<I3gmm>(new I3gmm(
			Vector(prior, d), Matrix(prior + d, d),
			conf[0], conf[1], conf[2], conf[3], conf[4]));
		break;
	case 9:
		plhs[0] = convertPtr2Mat<I3gmm>(new I3gmm(
			Vector(prior, d), Matrix(prior + d, d),
			conf[0], conf[1], conf[2], conf[3], conf[4],
			conf[5], conf[6], conf[7], conf[8]));
		break;
	default:
		mexErrMsgTxt("New: no matching configuration");
		break;
	}
	set_isI3gmm(plhs[0], isI3gmm);
}

void add_data(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], I3gmm *obj) {
	// Check parameters
	if (nrhs < 3) {
		mexErrMsgIdAndTxt("I3gmm:add_data", "3 or 4 inputs required.");
	}
	// Get dataset
	vector<Vector> dataset;	get_dataset(prhs[2], dataset);
	switch (nrhs)
	{
	case 3: // No weights
		obj->add_data(dataset);
		break;
	case 4: // With weights
		obj->add_data(dataset, mxGetPr(prhs[3]));
		break;
	}
	return;
}

void add_data(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], DP<Vector> *obj) {
	// Check parameters
	if (nrhs < 3) {
		mexErrMsgIdAndTxt("Igmm:add_data", "3 inputs required.");
	}
	vector<Vector> dataset;	get_dataset(prhs[2], dataset);
	obj->add_data(dataset);
	return;
}

template <class T>
void cluster_gibbs(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], T *i3gmm_obj) {
	switch (nrhs)
	{
	case 3:
		i3gmm_obj->cluster_gibbs(mxGetScalar(prhs[2]));
		break;
	case 4:
		i3gmm_obj->cluster_gibbs(mxGetScalar(prhs[2]), mxGetScalar(prhs[3]));
		break;
	case 5:
		i3gmm_obj->cluster_gibbs(mxGetScalar(prhs[2]), mxGetScalar(prhs[3]), mxGetScalar(prhs[4]));
		break;
	case 6:
		i3gmm_obj->cluster_gibbs(mxGetScalar(prhs[2]), mxGetScalar(prhs[3]), mxGetScalar(prhs[4]), mxArrayToString(prhs[5]));
		break;
	default:
		mexErrMsgIdAndTxt("I3gmm:cluster_gibbs", "Need 3 to 6 inputs. (sweeps, burnin, sample, logfname)");
		break;
	}
	plhs[0] = mxCreateDoubleMatrix(1, mxGetScalar(prhs[2]), mxREAL);
	copyDouble(plhs[0], i3gmm_obj->logLikeHistory);
	// Get sample
	if (i3gmm_obj->sample_labels.r > 0) {
		plhs[1] = mxCreateDoubleMatrix(i3gmm_obj->sample_labels.m, i3gmm_obj->sample_labels.r, mxREAL);
		copyDouble(plhs[1], i3gmm_obj->sample_labels.data);
	}
	return;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Get the command string
	char cmd[64];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");

	// New
	if (!strcmp("new", cmd)) {
		new_i3gmm(nlhs, plhs, nrhs, prhs);
		return;
	}

	// Check there is a second input, which should be the class instance handle
	if (nrhs < 2)
		mexErrMsgTxt("Second input should be a class instance handle.");

	// Delete
	if (!strcmp("delete", cmd)) {
		// Destroy the C++ object
		if (get_isI3gmm(prhs[1]))
			destroyObject<I3gmm>(prhs[1]);
		else
			destroyObject<DP<Vector>>(prhs[1]);
		// Warn if other commands were ignored
		if (nlhs != 0 || nrhs != 2)
			mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
		return;
	}

	// Get the class instance pointer from the second input
	I3gmm *i3gmm_obj;
	DP<Vector> *igmm_obj;
	if (get_isI3gmm(prhs[1]))
		i3gmm_obj = convertMat2Ptr<I3gmm>(prhs[1]);
	else
		igmm_obj = convertMat2Ptr<DP<Vector>>(prhs[1]);


	// Call the various class methods
	// Add unlabeled data
	if (!strcmp("add_data", cmd)) {
		if (get_isI3gmm(prhs[1]))
			add_data(nlhs, plhs, nrhs, prhs, i3gmm_obj);
		else
			add_data(nlhs, plhs, nrhs, prhs, igmm_obj);
		return;
	}
	// Add training data and labels.
	if (!strcmp("add_prior", cmd)) {
		// Check parameters
		if (nrhs < 4)
			mexErrMsgIdAndTxt("I3gmm:add_prior", "Four inputs required. (trainning data, labels).");
		// Get dataset
		vector<Vector> dataset;	get_dataset(prhs[2], dataset);
		Vector labels(mxGetPr(prhs[3]), mxGetNumberOfElements(prhs[3]));
		if (get_isI3gmm(prhs[1]))
			i3gmm_obj->add_prior(dataset, labels);
		else
			igmm_obj->add_prior(dataset, labels);
		return;
	}
	// Adjust weights for data points
	if (!strcmp("adjust_weights", cmd)) {
		// Check parameters
		if (nrhs != 3)
			mexErrMsgIdAndTxt("I3gmm:adjust_weights", "Three inputs required.");
		if (!get_isI3gmm(prhs[1]))
			mexErrMsgIdAndTxt("I3gmm:adjust_weights", "Only for i3gmm, not for igmm.");
		if (mxGetNumberOfElements(prhs[2]) != i3gmm_obj->allcusts.size())
			mexErrMsgIdAndTxt("I3gmm:adjust_weights:weights", "Numel of Weights must match the number of data points.");
		i3gmm_obj->adjust_weights(mxGetPr(prhs[2]));
		return;
	}
	// Do gibbs sweeps
	if (!strcmp("cluster_gibbs", cmd)) {
		if (get_isI3gmm(prhs[1]))
			cluster_gibbs(nlhs, plhs, nrhs, prhs, i3gmm_obj);
		else
			cluster_gibbs(nlhs, plhs, nrhs, prhs, igmm_obj);
		return;
	}
	// Calculate likelihood for each training point
	if (!strcmp("prior_like", cmd)) {
		if (!get_isI3gmm(prhs[1]))
			mexErrMsgIdAndTxt("I3gmm:adjust_weights", "Only for i3gmm, not for igmm.");
		i3gmm_obj->prior_llike();
		return;
	}
	// Get current labels
	if (!strcmp("get_current_labels", cmd)) {
		// Check parameters
		Matrix label;
		if (get_isI3gmm(prhs[1]))
			label = i3gmm_obj->gen_labels(i3gmm_obj->allcusts);
		else
			label = igmm_obj->gen_labels(igmm_obj->ordered_custs);
		plhs[0] = mxCreateDoubleMatrix(label.m, label.r, mxREAL);
		copyDouble(plhs[0], label.data);
		return;
	}
	// Get hyper parameters
	if (!strcmp("get_hyperparams", cmd)) {
		if (!get_isI3gmm(prhs[1]))
			mexErrMsgIdAndTxt("I3gmm:get_hyperparams", "Only for i2gmm or i3gmm.");
		int dim = HyperParams::mu0.n;
		mxArray *cell_array_ptr = mxCreateCellMatrix(5, 1);
		mxArray *mu0 = mxCreateDoubleMatrix(dim, 1, mxREAL);
		mxArray *psi0 = mxCreateDoubleMatrix(dim, dim, mxREAL);
		mxArray *kappa0 = mxCreateDoubleScalar(HyperParams::kappa0);
		mxArray *kappa1 = mxCreateDoubleScalar(HyperParams::kappa1);
		mxArray *m = mxCreateDoubleScalar(HyperParams::m);
		copyDouble(mu0, HyperParams::mu0.data);
		copyDouble(psi0, HyperParams::psi0.data);
		mxSetCell(cell_array_ptr, 0, mu0);
		mxSetCell(cell_array_ptr, 1, psi0);
		mxSetCell(cell_array_ptr, 2, kappa0);
		mxSetCell(cell_array_ptr, 3, kappa1);
		mxSetCell(cell_array_ptr, 4, m);
		plhs[0] = cell_array_ptr;
		return;
	}
	// Get likelihood for all customers
	if (!strcmp("get_likelihood", cmd)) {
		mxArray *llike = mxCreateDoubleMatrix(i3gmm_obj->allcusts.size(), 1, mxREAL);
		if (get_isI3gmm(prhs[1])) {
			copyDouble(llike, i3gmm_obj->likelihood().data);
		}
		else {
			mexErrMsgIdAndTxt("I3gmm::get_likelihood", "No implement for igmm");
		}
		plhs[0] = llike;
		return;
	}

	// Got here, so command not recognized
	mexErrMsgTxt("Command not recognized.");
}
