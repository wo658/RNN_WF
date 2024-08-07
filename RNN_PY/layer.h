#ifndef LAYER
#define LAYER

#include <vector>


class layer {

private:



public:
	//virtual void init();
	std::vector <double> value;
	std::vector <double> activate_value;

	std::vector<double> get_Value();
	std::vector<double> get_ActivateValue();

	layer();
};
class in_layer : public layer {
public:
	in_layer(int num);

	int node_num;
};
class out_layer : public layer {
public:
	out_layer(int num);

	int node_num;
};
class hidden_layer : public layer {

public:
	hidden_layer(int pre_num, int num, int next_num);
	
	void init();

	std::vector <std::vector <double>> w_to_h;
	std::vector <std::vector <double>> h_to_h;
	std::vector <std::vector <double>> h_to_o;
	std::vector <double> bias;
	int node_pre_num, node_next_num , node_num;



};

#endif 