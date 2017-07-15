#include "PID.h"
#include <iostream>
using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double in_Kp, double in_Ki, double in_Kd) {
	Kp = in_Kp;
	Ki = in_Ki;
	Kd = in_Kd;

	p_error = 0;
	i_error = 0;
	d_error = 0;
}

void PID::UpdateError(double cte) {
	cout << cte << endl;
	cout << "Parms: \n" << 
		"P: " << Kp << " P error: " << p_error << 
		"\n D: " << Kd << " D error: " << d_error << 
		"\n I: "<< Ki << " I error: " << i_error << endl;
	d_error = cte - p_error; // p_error = previous_error
	p_error = cte;
	i_error += cte;
}

double PID::TotalError() {
	return Kp * p_error + Ki * i_error + Kd * d_error;
}

