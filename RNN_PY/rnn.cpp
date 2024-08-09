#include "rnn.h"
#include <iostream>

double tanh_function(double x) {
	return std::tanh(x); 
}
double tanh_derivative(double x) {
	double tanh_x = tanh_function(x); 
	return 1.0 - tanh_x * tanh_x; 
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



void rnn::backpropagation( std::vector<std::vector<double>>& train_x, std::vector<std::vector<double>>& train_y) {
    double learning_rate = 0.0001;
    const int num_hidden_units = 30;

    std::vector<std::vector<double>> deltas(train_x.size(), std::vector<double>(num_hidden_units, 0.0));

    // Iterate backward through timesteps
    for (int i = train_x.size() - 1; i >= 0; i--) {
        // Output layer error
        double output_error = train_y[i][0] - outputs[i];
        std::vector<double> output_deltas(num_hidden_units, 0.0);
        output_deltas[0] = output_error;

        // Backpropagate error to hidden layer
        for (int j = 0; j < num_hidden_units; j++) {
            double delta = output_error * h_layer.h_to_o[j][0] * tanh_derivative(h_states[i][j]);
            deltas[i][j] = delta;

            // Update Hidden to Output Weights
            h_layer.h_to_o[j][0] -= learning_rate * delta * h_states[i][j];
        }

        // Hidden Layer(s) Error and Delta Calculation
        for (int j = 0; j < num_hidden_units; j++) {
            double delta = 0.0;
            if (i < train_x.size() - 1) { // Not the last timestep
                for (int k = 0; k < num_hidden_units; k++) {
                    delta += deltas[i + 1][k] * h_layer.h_to_h[j][k] * tanh_derivative(h_states[i][j]);
                }
            } else {
                // Handle the last timestep
                delta = output_error * tanh_derivative(h_states[i][j]);
            }

            // Update Input to Hidden Weights
            for (int k = 0; k < train_x[0].size(); k++) {
                h_layer.w_to_h[k][j] -= learning_rate * delta * train_x[i][k];
            }

            // Update Hidden to Hidden Weights
            if (i > 0) { // Not the first timestep
                for (int k = 0; k < num_hidden_units; k++) {
                    double pre_node = h_states[i - 1][k];
                    h_layer.h_to_h[k][j] -= learning_rate * delta * pre_node;
                }
            }
        }
    }
}


