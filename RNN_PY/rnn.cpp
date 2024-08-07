#include "rnn.h"
#include <iostream>
double tanh(double x) {
	double ex = std::exp(x);
	double ex_inv = std::exp(-x);
	return (ex - ex_inv) / (ex + ex_inv);
}
rnn::rnn(hidden_layer& h, out_layer& o, in_layer& i) : h_layer(h), o_layer(o), i_layer(i) {}

double rnn::feedforward(std::vector<double>& train) {
	std::vector<double> h(30, 0.0); // ���� ���� �ʱ�ȭ (30���� ���� ����)
	double tmp;
	double output;

	// ���� ���� ���� �ʱ�ȭ
	h_layer.h_states.clear();
	h_layer.h_states.push_back(h); // �ʱ� ���� ����

								   // ������
	for (int t = 0; t < train.size(); t++) {
		double input = train[t];

		// ���� ���� ������Ʈ
		std::vector<double> h_new(30, 0.0);
		for (int j = 0; j < 30; j++) {
			tmp = 0.0;
			tmp += h_layer.w_to_h[j][0] * input; // �Է��� ���
			if (t > 0) { // t�� 0�� �ƴ� ���� h_prev ���
				tmp += h_layer.h_to_h[j][j] * h[j]; // ���� ���� ���¸� ���
			}
			h_new[j] = tanh(tmp); 
		}
		h = h_new;
		h_layer.h_states.push_back(h); 

									   // �� �� TimeStep�� �����ϸ� ��� ���
		if (t == train.size() - 1) {
			output = 0.0;
			for (int j = 0; j < 30; j++) {
				output += h_layer.h_to_o[j][0] * h[j]; // h_to_o ���
			}
		}
	}

	return output;
}


void rnn::backpropagation() {

}

