#include "rnn.h"
#include <iostream>

double tanh(double x) {
	double ex = std::exp(x);
	double ex_inv = std::exp(-x);
	return (ex - ex_inv) / (ex + ex_inv);
}
double tanh_derivative(double x) {
	return 1.0 - x * x; // tanh'(x) = 1 - tanh^2(x)
}

rnn::rnn(hidden_layer& h, out_layer& o, in_layer& i , int size) : h_layer(h), o_layer(o), i_layer(i),train_size(size) {


	// time step 단위로 저장되는 변수
	h_states.resize(train_size);
	for (int i = 0; i < h_states.size(); i++)
		h_states.resize(h_layer.node_num);
	at.resize(train_size);
	for (int i = 0; i < at.size(); i++)
		at[i].resize(h_layer.node_num);
	outputs.resize(train_size);

	w_ih.resize(this->h_layer.node_pre_num);
	for (int i = 0; i < this->h_layer.node_pre_num; i++)
		w_ih[i].resize(this->h_layer.node_num);
	w_hh.resize(this->h_layer.node_num);
	for (int i = 0; i < this->h_layer.node_num; i++)
		w_hh[i].resize(this->h_layer.node_num);
	w_oh.resize(this->h_layer.node_num);
	for (int i = 0; i < this->h_layer.node_num; i++)
		w_oh[i].resize(this->h_layer.node_next_num);




}



void rnn::feedforward(std::vector<double>& train, int timestep) {
	static std::vector<double> prev_hidden_value(30, 0.0);  // 이전 Layer 의 Hidden 정보 Static 변수
	static bool is_initialized = false;

	if (!is_initialized) {
		for (int j = 0; j < 30; j++) {
			h_layer.bias[j] = 0.0;
		}
		o_layer.bias[0] = 0.0;
		is_initialized = true;
	}

	// Ensure h_states and outputs are large enough
	if (timestep >= h_states.size()) {
		h_states.resize(timestep + 1, std::vector<double>(30, 0.0));
	}
	if (timestep >= outputs.size()) {
		outputs.resize(timestep + 1, 0.0);
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
		at[timestep][j] = h_layer.value[j];
		h_layer.activate_value[j] = tanh(h_layer.value[j]);
	}
	// at[timestep][j] 






	// hidden to out
	o_layer.value[0] = 0.0;
	for (int i = 0; i < 30; i++) {
		o_layer.value[0] += h_layer.h_to_o[i][0] * h_layer.activate_value[i];
	}
	o_layer.value[0] += o_layer.bias[0];

	// Update previous hidden values and save states
	prev_hidden_value = h_layer.activate_value;
	h_states[timestep] = prev_hidden_value;
	outputs[timestep] = o_layer.value[0];
}



void rnn::backpropagation(std::vector<std::vector<double>>& train_x, std::vector<std::vector<double>>& train_y) {

	double learning_rate = 0.0001; // 학습률 설정

								 // Timestep을 역으로 되짚으며 BackPropagation
	for (int i = train_x.size() - 1; i >= 0; i--) {

		std::vector<double> deltas(30, 0.0);

		// 1. Output Layer Error and Delta Calculation
		for (int j = 0; j < 30; j++) {
			double output_error = train_y[i][0] - outputs[i]; // 출력과 실제 값의 차이
			double delta = output_error * tanh_derivative(outputs[i]); // 출력층의 델타

			deltas[j] = delta; // Backpropagation을 위해 델타값 저장

							   // Update: Hidden to Output Weights
			h_layer.h_to_o[j][0] -= learning_rate * delta * h_states[i][j];
		}

		// 2. Hidden Layer(s) Error and Delta Calculation
		for (int j = 0; j < 30; j++) {
			double delta = deltas[j];

			if (i != train_x.size() - 1) {  // 마지막 타임스텝이 아닌 경우, 다음 타임스텝의 델타값을 고려하여 계산
				for (int k = 0; k < 30; k++) {
					delta += deltas[k] * h_layer.h_to_h[j][k] * tanh_derivative(at[i][j]);
				}
			}

			deltas[j] = delta;

			// Update 1: Input to Hidden Weights
			for (int k = 0; k < train_x[0].size(); k++) {
				h_layer.w_to_h[k][j] -= learning_rate * delta * train_x[i][k];
			}

			// Update 2: Hidden to Hidden Weights
			for (int k = 0; k < 30; k++) {
				double pre_node = (i != 0) ? h_states[i - 1][k] : 0.0;
				h_layer.h_to_h[k][j] -= learning_rate * delta * pre_node;
			}

		}
	}
}

