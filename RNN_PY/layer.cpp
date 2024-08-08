#include "layer.h"
double random_number() {
	return (rand() % 20 - 10) / 10.0;
}
layer::layer() {
	

}


std::vector <double> layer::get_ActivateValue() {
	return activate_value;
}
std::vector <double> layer::get_Value() {
	return value;
}

out_layer::out_layer(int num) : node_num(num) {
	value.resize(num);
	activate_value.resize(num);
	bias.resize(num);
}
in_layer::in_layer(int num) : node_num(num) {
	value.resize(num);
	activate_value.resize(num);
}
hidden_layer::hidden_layer(int pre_num,int num,int next_num):node_pre_num(pre_num), node_num(num), node_next_num(next_num) {
	// weight 
	// input -> hidden , hidden -> out , hidden -> hidden ( timeline ) , 
	// 행이 w , 열이 h
	w_to_h.resize(pre_num);
	for (int i = 0; i < pre_num; i++)
		w_to_h[i].resize(num);
	h_to_h.resize(num);
	for (int i = 0; i < num; i++)
		h_to_h[i].resize(num);
	h_to_o.resize(num);
	for (int i = 0; i < node_num; i++)
		h_to_o[i].resize(next_num);
	value.resize(num);
	bias.resize(num);
	activate_value.resize(num);
	for (int i = 0; i < num;i++)
		h_new.push_back(0);

	// .....


}
void hidden_layer::init() {
	for (int i = 0; i < node_pre_num; i++)
		for (int j = 0; j < node_num; j++)
			w_to_h[i][j] = random_number();

	for (int i = 0; i < node_num; i++)
		for (int j = 0; j < node_num; j++)
			h_to_h[i][j] = random_number();

	for (int i = 0; i < node_num; i++)
		for (int j = 0; j < node_next_num; j++)
			h_to_o[i][j] = random_number();
	for (int i = 0; i < node_num; i++)
		bias[i] = 3.0;
}
