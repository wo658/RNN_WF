#include "rnn.h"
#include <iostream>
double tanh(double x) {
	double ex = std::exp(x);
	double ex_inv = std::exp(-x);
	return (ex - ex_inv) / (ex + ex_inv);
}
rnn::rnn(hidden_layer& h, out_layer& o, in_layer& i) : h_layer(h), o_layer(o), i_layer(i) {}

double rnn::feedforward(std::vector<double>& train) {
	std::vector<double> h(30, 0.0); // 히든 상태 초기화 (30개의 히든 유닛)
	double tmp;
	double output;

	// 히든 상태 저장 초기화
	h_layer.h_states.clear();
	h_layer.h_states.push_back(h); // 초기 상태 저장

								   // 순전파
	for (int t = 0; t < train.size(); t++) {
		double input = train[t];

		// 히든 상태 업데이트
		std::vector<double> h_new(30, 0.0);
		for (int j = 0; j < 30; j++) {
			tmp = 0.0;
			tmp += h_layer.w_to_h[j][0] * input; // 입력을 고려
			if (t > 0) { // t가 0이 아닐 때만 h_prev 사용
				tmp += h_layer.h_to_h[j][j] * h[j]; // 이전 히든 상태를 고려
			}
			h_new[j] = tanh(tmp); 
		}
		h = h_new;
		h_layer.h_states.push_back(h); 

									   // 맨 끝 TimeStep에 도달하면 출력 계산
		if (t == train.size() - 1) {
			output = 0.0;
			for (int j = 0; j < 30; j++) {
				output += h_layer.h_to_o[j][0] * h[j]; // h_to_o 사용
			}
		}
	}

	return output;
}


void rnn::backpropagation() {

}

