#ifndef RNN
#define RNN

#include "layer.h"

class rnn {
public:
	rnn(hidden_layer& h, out_layer& o, in_layer& i , int train_size);
	

	hidden_layer h_layer;
	out_layer o_layer;
	in_layer i_layer;
	int train_size;


	// w 값 계산을 위한 기울기 저장
	std::vector<std::vector<double>> w_ih;
	std::vector<std::vector<double>> w_hh;
	std::vector<std::vector<double>> w_oh;



	std::vector<std::vector<double>> h_states;
	std::vector<double> outputs;
	std::vector<std::vector<double>> at;

	void feedforward(std::vector < double >& train  , int timestep);
	void backpropagation(std::vector<std::vector < double >>& train_x , std::vector<std::vector < double >>& train_y);

};

#endif 