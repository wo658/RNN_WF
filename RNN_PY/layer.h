#ifndef LAYER
#define LAYER

#include <vector>


class layer {

private:
	std::vector <double> value;
	std::vector <double> activate_value;


public:
	//virtual void init();

	std::vector<double> get_Value();
	std::vector<double> get_ActivateValue();

	layer();
};
class in_layer : public layer {

};
class out_layer : public layer {
	out_layer(int num);
	double bias;
	int node_num;
};
class hidden_layer : public layer {
	hidden_layer(int pre_num, int num, int next_num);
	double bias;
	void init();

	std::vector <std::vector <double>> w_to_h;
	std::vector <std::vector <double>> h_to_h;
	std::vector <std::vector <double>> h_to_o;


	int node_pre_num, node_next_num , node_num;


};

#endif 