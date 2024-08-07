#ifndef RNN
#define RNN

#include "layer.h"

class rnn {
public:
	rnn(hidden_layer& h, out_layer& o, in_layer& i);
	

	hidden_layer h_layer;
	out_layer o_layer;
	in_layer i_layer;

	
	double feedforward(std::vector < double >& train );
	void backpropagation();

};

#endif 