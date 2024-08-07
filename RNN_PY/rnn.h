#ifndef RNN
#define RNN

#include "layer.h"

class rnn {
public:
	rnn(hidden_layer& h, out_layer& o, in_layer& i);
	

	hidden_layer h_layer;
	out_layer o_layer;
	in_layer i_layer;

	// data ÀÇ °³¼ö
	void feedforward(std::vector < std::vector<double >> train);
	void backpropagation();
	double loss_cal();


};

#endif 