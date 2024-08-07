#include "rnn.h"

double tanh(double x) {
	double ex = std::exp(x);
	double ex_inv = std::exp(-x);
	return (ex - ex_inv) / (ex + ex_inv);
}
rnn::rnn(hidden_layer& h, out_layer& o, in_layer& i) : h_layer(h), o_layer(o), i_layer(i) {}

void rnn::feedforward(std::vector<std::vector<double>> train) {



}

void rnn::backpropagation() {

}
double rnn::loss_cal() {
	return 0;
}