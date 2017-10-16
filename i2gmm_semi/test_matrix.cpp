#include <iostream>
#include <Eigen/Dense>
#include "Matrix.h"
#include "DebugUtils.h"
using Eigen::MatrixXd;

int main()
{
	debugMode(1);
	MatrixXd m1 = MatrixXd::Random(4, 4);
	m1 = (m1 + MatrixXd::Constant(4, 4, 1)) * 0.5;
	Matrix m2(4);
	m2.zero();
	m2.random();
	//m2.eye();
	init_buffer(1, 4);

	cout << "Eigen:" << endl;
	cout << m1 << endl;
	step();
	for (int i = 0; i < 10; i++){
		m1 = m1*m1;
	}
	step();
	cout << m1 << endl;

	cout << "FastMat:" << endl;
	cout << m2 << endl;
	step();
	for (int i = 0; i < 10000; i++){
		m2 = m2*m2;
	}
	step();
	cout << m2 << endl;

	system("pause");
}