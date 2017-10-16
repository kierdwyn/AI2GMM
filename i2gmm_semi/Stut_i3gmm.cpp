#include "Stut_i3gmm.h"
#include "DP.h"
#include "Wishart.h"


Vector HyperParams::mu0;
Vector HyperParams::mu0ML;
Matrix HyperParams::psi0;
Matrix HyperParams::psi0ML;
double HyperParams::kappa0;
double HyperParams::kappa1;
int HyperParams::m;
Matrix HyperParams::last_psi0;
double HyperParams::last_kappa0;
double HyperParams::last_kappa1;
int HyperParams::last_m;
Vector HyperParams::mu00;
Matrix HyperParams::sigma0_inv;
Matrix HyperParams::sigma0;
double HyperParams::c1;
double HyperParams::c2;
double HyperParams::alpha0;
double HyperParams::beta0;
double HyperParams::alpha1;
double HyperParams::beta1;
double HyperParams::kap1over0;

Vector HyperParams::mx_all;
Matrix HyperParams::sx_all;
double HyperParams::nx_all;


HyperParams::HyperParams(Vector mu0, Matrix sigma0, double m,
	double kappa0, double kappa1) {
	HyperParams::init(mu0, sigma0, m, kappa0, kappa1);
}

HyperParams::HyperParams(Vector mu0, Matrix sigma0, double m,
	double c1, double c2, double alpha0, double beta0,
	double alpha1, double beta1) {
	HyperParams::init(mu0, sigma0, m, c1, c2, alpha0, beta0, alpha1, beta1);
}

void HyperParams::init(Vector mu0, Matrix sigma0, double m,
	double kappa0, double kappa1){
	HyperParams::mu0 = Vector(mu0);
	HyperParams::psi0 = Matrix(sigma0);
	HyperParams::kappa0 = kappa0;
	HyperParams::kappa1 = kappa1;
	HyperParams::m = m;
}

void HyperParams::init(Vector mu0, Matrix sigma0, double m,
	double c1, double c2, double alpha0, double beta0,
	double alpha1, double beta1){
	HyperParams::mu0 = Vector(mu0);
	//HyperParams::psi0 = Matrix(sigma0/(c2 + mu0.n + 1));
	HyperParams::psi0 = Matrix(sigma0 * (c2 - mu0.n));
	HyperParams::kappa0 = (alpha0 - 1) / beta0;
	HyperParams::kappa1 = (alpha1 - 1) / beta1;
	HyperParams::m = m;
	HyperParams::last_psi0 = Matrix(psi0);
	HyperParams::last_kappa0 = kappa0;
	HyperParams::last_kappa1 = kappa1;
	HyperParams::last_m = HyperParams::m;
	HyperParams::mu00 = Vector(mu0);
	HyperParams::sigma0_inv = Matrix(sigma0.inverse());
	HyperParams::sigma0 = sigma0;
	HyperParams::c1 = c1;
	HyperParams::c2 = c2;
	HyperParams::alpha0 = alpha0;
	HyperParams::alpha1 = alpha1;
	HyperParams::beta0 = beta0;
	HyperParams::beta1 = beta1;
	HyperParams::kap1over0 = kappa1 / kappa0;
}

template <class Iter>
void HyperParams::renew_hyper_params(Iter first, Iter last, int psi0choice){
	int K = 0, d = mu0.n;
	Vector mu0_1(d); mu0_1.zero();
	Matrix sum_sigma_hinv(d); sum_sigma_hinv.zero();
	double kappa1_1 = 0, kappa1_2 = 0, kappa0_1 = 0;
	Matrix sx_all1 = sx_all;
	Vector mx_all1 = mx_all;
	int nx_all1 = nx_all;

	for (Iter citer = first; citer != last; ++citer){
		DP<Vector> *cluster = dynamic_cast<DP<Vector>*>(*citer);
		StutGlobal3 *c_dist = dynamic_cast<StutGlobal3*>(cluster->get_dist());
		if (c_dist->sum_n < mu0.n) { // Skip clusters don't have enough points.
			if (psi0choice > 1) {
				sx_all1 -= c_dist->sum_scatter;
				for (Table<Vector> *t : cluster->tables) {
					StutLocal3 *l_dist = dynamic_cast<StutLocal3*>(t->H);
					mx_all1 -= l_dist->mean*l_dist->n_points;
					nx_all1 -= l_dist->n_points;
				}
			}
			continue;
		}
		if (psi0choice > 1){
			for (Customer<Vector>* cust : cluster->unassigned_custs) {
				sx_all1 -= cust->data >> cust->data;
				mx_all1 -= cust->data;
				nx_all1--;
			}
		}

		// Update mu_kl
		c_dist->sum_mu_kl.zero();
		c_dist->sum_scatter_kl.zero();
		for (Table<Vector> *t : cluster->tables) {
			StutLocal3 *l_dist = dynamic_cast<StutLocal3*>(t->H);
			l_dist->update_hidden_vars();
			c_dist->update_hidden_vars(l_dist, 1);
		}
		// Update mu_h (mu_k),sigma_k
		c_dist->update_hidden_vars();

		Matrix sigma_hinv = c_dist->sigma_h.inverse();
		// Update kappa1_2
		for (Table<Vector> *t : cluster->tables) {
			StutLocal3 *l_dist = dynamic_cast<StutLocal3*>(t->H);
			Vector diff = l_dist->mu_kl - c_dist->mu_h;
			kappa1_2 += diff * (sigma_hinv * diff);
		}
		// Update kappa0_1, kappa1_1
		Vector diff = c_dist->mu_h - mu0;
		kappa0_1 += diff * (sigma_hinv * diff);
		kappa1_1 += c_dist->n_clusters;
		sum_sigma_hinv += sigma_hinv;
		mu0_1 += (sigma_hinv * c_dist->mu_h);
		K++;
	}
	kappa1 = (2 * (alpha1 - 1) + d*kappa1_1) / (2 * beta1 + kappa1_2);
	kappa0 = (2 * (alpha0 - 1) + d*K) / (2 * beta0 + kappa0_1);
	kappa0 = 1e-4 > kappa0 ? 1e-4 : kappa0;
	kappa1 = 1e-4 > kappa1 ? 1e-4 : kappa1;
	Vector diff = mu0 - mu00;
	psi0 = (sigma0_inv + (diff >> diff)*c1 + sum_sigma_hinv).inverse() * (c2 - d + m*K) + eye(d)*1e-4;
	mu0 = (psi0*mu00*c1 + mu0_1*kappa0) / (psi0*c1 + sum_sigma_hinv*kappa0);

	double a1;
	if (psi0choice == 2 || psi0choice == 4) {
		a1 = (m + 1) / (1 + 1 / kappa0 + 1 / kappa1);
		mu0ML = mx_all1 / nx_all1;
		psi0ML = (sx_all1 / nx_all1 - (mu0ML >> mu0ML))*a1 + eye(d)*1e-4;
		if (psi0choice == 4) {
			mu0 = mu0ML;
			psi0 = psi0ML;
		}
	}
	if (psi0choice == 3) {
		a1 = (m + 1 + nx_all1) / (1 + 1 / kappa0 + 1 / kappa1);
		mu0ML = mx_all1 / nx_all1;
		psi0ML = (sx_all1*a1 / nx_all1 - (mu0ML >> mu0ML)*a1) + eye(d)*1e-4;
	}
}

//template <class Iter>
//void HyperParams::renew_hyper_params(Iter first, Iter last) {
//	int K = 0, d = mu0.n;
//	double kappa1_1 = 0, kappa1_2 = 0, kappa0_1 = 0;
//	Vector mu0_1(d); mu0_1.zero();
//	Matrix sum_sigma_hinv(d); sum_sigma_hinv.zero();
//
//	/*Matrix sum_scatter(d); sum_scatter.zero();
//	double sum_n=0;*/
//
//	for (Iter citer = first; citer != last; ++citer) {
//		DP<Vector> *cluster = dynamic_cast<DP<Vector>*>(*citer);
//		StutGlobal3 *c_dist = dynamic_cast<StutGlobal3*>(cluster->get_dist());
//		if (c_dist->sum_n < mu0.n) // Skip clusters don't have enough points.
//			continue;
//
//		// Update mu_kl
//		c_dist->sum_mu_kl.zero();
//		c_dist->sum_scatter_kl.zero();
//		for (Table<Vector> *t : cluster->tables) {
//			StutLocal3 *l_dist = dynamic_cast<StutLocal3*>(t->H);
//			l_dist->update_hidden_vars();
//			c_dist->update_hidden_vars(l_dist, 1);
//		}
//		// Update mu_h (mu_k),sigma_k
//		c_dist->update_hidden_vars();
//
//		Matrix sigma_hinv = c_dist->sigma_h.inverse();
//		for (Table<Vector> *t : cluster->tables) {
//			StutLocal3 *l_dist = dynamic_cast<StutLocal3*>(t->H);
//			// Update kappa1_2
//			Vector diff = l_dist->mu_kl - c_dist->mu_h;
//			kappa1_2 += diff * (sigma_hinv * diff);
//		}
//		// Update kappa0_1, kappa1_1
//		Vector diff = c_dist->mu_h - mu0;
//		kappa0_1 += diff * (sigma_hinv * diff);
//		kappa1_1 += c_dist->n_clusters;
//		sum_sigma_hinv += sigma_hinv;
//		mu0_1 += (sigma_hinv * c_dist->mu_h);
//		K++;
//
//		/*sum_scatter += c_dist->sum_scatter;
//		sum_n += c_dist->sum_n;*/
//	}
//	kappa1 = (2 * (alpha1 - 1) + d*kappa1_1) / (2 * beta1 + kappa1_2);
//	kappa0 = (2 * (alpha0 - 1) + d*K) / (2 * beta0 + kappa0_1);
//	kappa0 = 1e-2 > kappa0 ? 1e-2 : kappa0;
//	kappa1 = 1e-2 > kappa1 ? 1e-2 : kappa1;
//
//	double ks = harmean(kappa0, kappa1);
//	double a1 = (m + 1) / (1 + 1 / kappa0 + 1 / kappa1);
//	/*mu0 = (psi0*mu00*c1 + mu0_1*kappa0) / (psi0*c1 + sum_sigma_hinv*kappa0);*/
//
//	/*psi0 = (sum_scatter)*m / sum_n + eye(d)*1e-4;*/
//	mu0 = mx_all / nx_all;
//	psi0 = (sx_all*a1/nx_all - (mu0 >> mu0)*a1) + eye(d)*1e-4;
//	/*psi0 = ((sx_all - (mu0 >> mu0)*nx_all)*a1 + sigma0) /
//		(nx_all + c2 + d + 1) + eye(d)*1e-4;*/
//}

template<class Iter>
inline void HyperParams::update_global_stats(Iter first, Iter last) {
	sx_all = Matrix(d); sx_all.zero();
	mx_all = Vector(d); mx_all.zero();
	nx_all = 0;
	for (Iter citer = first; citer != last; ++citer) {
		Customer<Vector> *c = dynamic_cast<Customer<Vector>*>(*citer);
		sx_all += c->data >> c->data;
		mx_all += c->data;
		nx_all++;
	}
}


StutGlobal3::StutGlobal3(){
	init();
}

StutGlobal3::StutGlobal3(Vector mu0, Matrix psi0, double m,
	double kappa0, double kappa1)
	:HyperParams(mu0,psi0,m,kappa0,kappa1) {
	init();
}

StutGlobal3::StutGlobal3(Vector mu0, Matrix sigma0, double m,
	double c1, double c2, double alpha0, double beta0,
	double alpha1, double beta1)
	:HyperParams(mu0, sigma0, m, c1, c2, alpha0, beta0, alpha1, beta1){
	init();
}

double StutGlobal3::loglike(const Table<Vector> &t, int tablelike){
	double ret = 0;
	switch (tablelike)
	{
	case 0: // likelihood of all points belongs to empty component
	case 1:
	case 2:{ // likelihood of all points to its component after moving to current cluster
		StutLocal3 *dist = nullptr;
		if (tablelike) {
			dist = new StutLocal3(*dynamic_cast<StutLocal3*>(t.H));
			dist->change_global(this);
		}
		for (Customer<Vector> *x : t.custs) {
			if (tablelike > 1) ret += 0.5*Stut::likelihood(x->data) + 0.5*dist->likelihood(x->data) + log(x->weight);
			else if (tablelike) ret += dist->likelihood(x->data) + log(x->weight);
			else ret += Stut::likelihood(x->data) + log(x->weight);
		}
		delete dist;
	}
		break;
	//case 3: { // likelihood of the component's sufficient statistics conditioned on current cluster
	//	StutLocal3 *dist1 = dynamic_cast<StutLocal3*>(t.H);
	//	if (n_clusters > 1e-5) {
	//		Normal N(mu_h, sigma_h*(t.n_custs + kappa1) / (t.n_custs*kappa1));
	//		Wishart W(sigma_h, t.n_custs);
	//		if (t.n_custs > d)
	//			ret = N.likelihood(dist1->mean) + W.likelihood(dist1->scatter);
	//		else
	//			ret = N.likelihood(dist1->mean);

	//	}
	//	else {
	//		double df = m + t.n_custs + 1 - mu0.n;
	//		Stut st(mu0, (dist1->scatter + psi0)*(1 / t.n_custs + 1 / kappa1 + 1 / kappa0) / df, df);
	//		ret = st.likelihood(dist1->mean);
	//		if (t.n_custs > d)
	//			ret += dist1->scatter.chol().sumlogdiag()*(t.n_custs - mu0.n - 1)
	//				+ psi0.chol().sumlogdiag()*m + gamlnd(m + t.n_custs, mu0.n)
	//				- (dist1->scatter + psi0).chol().sumlogdiag()*(m + t.n_custs)
	//				- gamlnd(t.n_custs, mu0.n) - gamlnd(m, mu0.n);
	//	}
	//}
	//	break;
	default:
		break;
	}
	return ret;
}

void StutGlobal3::add_component(StutLocal3 *x_dist){
	update_statistics(x_dist, 1);
	update_stut();
}

void StutGlobal3::remove_component(StutLocal3 *x_dist){
	update_statistics(x_dist, -1);
	update_stut();
}

template <class Iter>
void StutGlobal3::add_all(Iter first, Iter last){
	for (Iter iter = first; iter != last; ++iter){
		Table<Vector> *t = dynamic_cast<Table<Vector>*>(*iter);
		StutLocal3 *x_dist = dynamic_cast<StutLocal3*>(t->H);
		update_statistics(x_dist, 1);
	}
	update_stut();
}

void StutGlobal3::reset(){
	n_clusters = 0;
	sum_weighted_n = 0;
	sum_n = 0;
	weighted_mean.zero();
	sum_scatter.zero();
}

void StutGlobal3::update_hidden_vars(){
	mu_h = (mu0*kappa0 + sum_mu_kl*kappa1) / (kappa0 + n_clusters*kappa1);
	Vector diff = mu_h - mu0;
	/*Matrix Smu0k = mu0 >> mu_h;
	/*Matrix sum_scatter_h = sum_scatter_kl * kappa1
		+ (Smu0k + Smu0k.transpose())*kappa0 - (mu_h >> mu_h)*(2 * kappa0 + n_clusters*kappa1);*/
	Matrix Smuhk = sum_mu_kl>>mu_h;
	Matrix sum_scatter_h = sum_scatter_kl - (Smuhk + Smuhk.transpose())
		+ (mu_h >> mu_h)*n_clusters;
	sigma_h = (psi0 + (diff >> diff)*kappa0 + sum_scatter_h*kappa1 + sum_scatter)
		/ (m + mu0.n + 2 + sum_n);
}

void StutGlobal3::update_hidden_vars(StutLocal3 *x_dist, int sign){
	sum_mu_kl += x_dist->mu_kl * sign;
	sum_scatter_kl += (x_dist->mu_kl >> x_dist->mu_kl) * sign;
}

void StutGlobal3::update_statistics(StutLocal3 *x_dist, int sign){
	n_clusters += sign * (x_dist->n_points / x_dist->n_points_unweighted);
	if (n_clusters == INFINITY || n_clusters == -INFINITY || !(n_clusters == n_clusters))
		PERROR("I3gmm:StutGlobal3:update_statistics: n_clusters infinity");
	double weighted_n = harmean(x_dist->n_points, kappa1);
	sum_weighted_n += sign * weighted_n;
	sum_n += sign * x_dist->n_points;
	weighted_mean += x_dist->mean * weighted_n * sign;
	sum_scatter += x_dist->scatter * sign;
}

void StutGlobal3::update_stut(int isML){ // TODO: combine with Stut. like set_statistics(mu, sigma, eta)
	kappa_s = (sum_weighted_n + kappa0)*kappa1
		/ (sum_weighted_n + kappa0 + kappa1);

	if (isML >= 2) {
		mu = (weighted_mean + mu0ML*kappa0) / (sum_weighted_n + kappa0);
		eta = m - mu0ML.n + 1 + sum_n - n_clusters;
		cholsigma = ((psi0ML + sum_scatter)*(kappa_s + 1) / (kappa_s*eta)).chol();
		if (isML == 3) {
			eta = m + 1 + nx_all;
			cholsigma = ((psi0ML + sum_scatter)*(kappa_s + 1) / (kappa_s*eta)).chol();
		}
	}
	else {
		mu = (weighted_mean + mu0*kappa0) / (sum_weighted_n + kappa0);
		eta = m - mu0.n + 1 + sum_n - n_clusters;
		cholsigma = ((psi0 + sum_scatter)*(kappa_s + 1) / (kappa_s*eta)).chol();
	}

	calculateNormalizer();
	if (normalizer == INFINITY || normalizer == -INFINITY || !(normalizer == normalizer)){
		PERROR("I3gmm:StutGlobal3:update_stut: normalizer infinity");
	}
}

void StutGlobal3::init(){
	int d = mu0.n;
	weighted_mean = Vector(d); sum_scatter = Matrix(d);
	weighted_mean.zero(); sum_scatter.zero();
	update_stut();

	sum_scatter_kl = Matrix(d); sum_scatter_kl.zero();
	sum_mu_kl = Vector(d); sum_mu_kl.zero();
	mu_h = mu0;
	sigma_h = psi0 / (m + mu0.n + 2);
	//sigma_hinv = Matrix(d); sigma_hinv.zero();
}


Stut3::Stut3()
{
	int d = mu0.n;
	mean.resize(d); mean.zero();
	scatter.resize(d, d); scatter.zero();
}

void Stut3::addData(const Vector & x, double weight)
{
	Vector diff = x - mean;
	/*scatter = (scatter + (diff >> diff) * weight / (n_points + weight))
		* n_points / (n_points + weight);*/
	scatter += (diff >> diff) * weight * n_points / (n_points + weight);
	mean = (mean*n_points + x*weight) / (n_points + weight);
	n_points += weight;
	n_points_unweighted += 1;

	update_stut(&x);
}

void Stut3::removeData(const Vector & x, double weight)
{
	if (n_points - weight > 0) {
		Vector diff = x - mean;
		scatter -= (diff >> diff)*weight*n_points / (n_points - weight);
		mean = (mean*n_points - x*weight) / (n_points - weight);
	}
	else {
		mean.zero();
		scatter.zero();
	}
	n_points -= weight;
	n_points_unweighted -= 1;

	update_stut(&x);
}

template<class Iter>
inline void Stut3::add_all(Iter first, Iter last)
{
	for (Iter iter = first; iter != last; ++iter) {
		Customer<Vector> *cust = dynamic_cast<Customer<Vector>*>(*iter);
		addData(cust->data, cust->weight);
	}
	update_stut(nullptr);
}

void Stut3::reset()
{
	n_points = 0;
	n_points_unweighted = 0;
	mean.zero();
	scatter.zero();
}


StutLocal3::StutLocal3(StutGlobal3 *stut) : g_dist(stut), Stut3()
{
	update_stut();
}

void StutLocal3::update_hidden_vars() {
	mu_kl = (g_dist->mu_h*g_dist->kappa1 + mean*n_points) /
		(g_dist->kappa1 + n_points);
}

//template <class Iter>
//void StutLocal3::add_all(Iter first, Iter last){
//	for (Iter iter = first; iter != last; ++iter){
//		Customer<Vector> *cust = dynamic_cast<Customer<Vector>*>(*iter);
//		addData(cust->data, cust->weight);
//	}
//	update_stut();
//}

void StutLocal3::update_stut(const Vector *x){
	Vector diff = mean - g_dist->mu;
	Matrix s_mu = (diff >> diff)*harmean(n_points, g_dist->kappa_s);

	double n_ks = n_points + g_dist->kappa_s;
	mu = (mean*n_points + g_dist->mu*g_dist->kappa_s) / n_ks;
	eta = g_dist->eta + n_points;
	Matrix chols = ((g_dist->psi0 + g_dist->sum_scatter + scatter + s_mu)
		* (n_ks + 1) / (n_ks*eta));
	cholsigma = chols.chol();
	calculateNormalizer();
	if (normalizer == INFINITY || normalizer == -INFINITY || !(normalizer == normalizer))
		PERROR("I3gmm:StutLocal3:update_stut: normalizer infinity");
}


void StutNIW::update_stut(const Vector *x){
	double n_ks = kappa0 + n_points;
	mu = (mu0*kappa0 + mean*n_points) / n_ks;
	eta = n_points + m - mu0.n + 1;

	if (false && x != nullptr) { // Perform rank 1 Cholesky update
		double lambda_n = (n_ks + 1) / (n_ks*eta);
		double lambda_n_1 = n_ks / ((n_ks - 1)*(eta - 1));
		cholsigma = cholsigma / sqrt(lambda_n_1);
		cholsigma = cholsigma.chol((*x - mu)* sqrt(n_ks / (n_ks - 1)));
		cholsigma = cholsigma * sqrt(lambda_n);
	}
	else {	// Perform normal cholesky decomposition
		Vector diff = mean - mu0;
		cholsigma = ((psi0 + scatter +
			(diff >> diff)*harmean(kappa0, n_points))
			* (n_ks + 1) / (n_ks*eta)).chol();
	}

	calculateNormalizer();
	if (normalizer == INFINITY || normalizer == -INFINITY || !(normalizer == normalizer))
		PERROR("I3gmm:StutNIW:update_stut: normalizer infinity");
}

template void HyperParams::renew_hyper_params<unordered_set<DP<Vector>*>::iterator>(
	unordered_set<DP<Vector>*>::iterator, unordered_set<DP<Vector>*>::iterator, int);
template void HyperParams::update_global_stats<vector<Customer<Vector>*>::iterator>(
	vector<Customer<Vector>*>::iterator, vector<Customer<Vector>*>::iterator);

template void StutGlobal3::add_all<unordered_set<Table<Vector>*>::iterator>(
	unordered_set<Table<Vector>*>::iterator, unordered_set<Table<Vector>*>::iterator);
template void StutGlobal3::add_all<vector<Table<Vector>*>::iterator>(
	vector<Table<Vector>*>::iterator, vector<Table<Vector>*>::iterator);

template void Stut3::add_all<unordered_set<Customer<Vector>*>::iterator>(
	unordered_set<Customer<Vector>*>::iterator, unordered_set<Customer<Vector>*>::iterator);
template void Stut3::add_all<vector<Customer<Vector>*>::iterator>(
	vector<Customer<Vector>*>::iterator, vector<Customer<Vector>*>::iterator);