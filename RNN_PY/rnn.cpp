#include "rnn.h"
#include <iostream>
double tanh(double x) {
	double ex = std::exp(x);
	double ex_inv = std::exp(-x);
	return (ex - ex_inv) / (ex + ex_inv);
}
rnn::rnn(hidden_layer& h, out_layer& o, in_layer& i) : h_layer(h), o_layer(o), i_layer(i) {}

double rnn::feedforward(std::vector<double>& train) {
	

	// input to hidden
	for (int j = 0; j < 30; j++) {
		for (int i = 0; i < train.size(); i++) {
			h_layer.value[j] += train[i] * h_layer.w_to_h[i][j];
		}
		h_layer.activate_value[j] = tanh(h_layer.value[j]);
	}

	// hidden to out
	for (int i = 0; i < 30; i++) {
		o_layer.value[0] += h_layer.h_to_o[i][0] * h_layer.activate_value[i];
	}

}


void rnn::backpropagation() {

}

