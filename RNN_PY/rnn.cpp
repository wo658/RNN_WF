#include "rnn.h"
#include <iostream>
double tanh(double x) {
	double ex = std::exp(x);
	double ex_inv = std::exp(-x);
	return (ex - ex_inv) / (ex + ex_inv);
}
rnn::rnn(hidden_layer& h, out_layer& o, in_layer& i) : h_layer(h), o_layer(o), i_layer(i) {}

double rnn::feedforward(std::vector<double>& train) {
    static std::vector<double> prev_hidden_value(30, 0.0);  // 이전 hidden state 값
    static bool is_initialized = false;

    if (!is_initialized) {
        for (int j = 0; j < 30; j++) {
            h_layer.bias[j] = 0.0;  
        }
        o_layer.bias[0] = 0.0; 
        is_initialized = true;
    }

    // input to hidden
    for (int j = 0; j < 30; j++) {
        h_layer.value[j] = 0.0; 
        for (int i = 0; i < train.size(); i++) {
            h_layer.value[j] += train[i] * h_layer.w_to_h[i][j];
        }

        for (int i = 0; i < 30; i++) {
            h_layer.value[j] += prev_hidden_value[i] * h_layer.h_to_h[i][j];
        }
        h_layer.value[j] += h_layer.bias[j];
        h_layer.activate_value[j] = tanh(h_layer.value[j]);
    }

    // hidden to out
    o_layer.value[0] = 0.0;  
    for (int i = 0; i < 30; i++) {
        o_layer.value[0] += h_layer.h_to_o[i][0] * h_layer.activate_value[i];
    }
    o_layer.value[0] += o_layer.bias[0];
    prev_hidden_value = h_layer.activate_value;

    return o_layer.value[0];
}



void rnn::backpropagation() {

}
