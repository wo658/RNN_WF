#include "layer.h"

layer::layer() {
	

}


std::vector <double> layer::get_ActivateValue() {
	return activate_value;
}
std::vector <double> layer::get_Value() {
	return value;
}

out_layer::out_layer(int num) : node_num(num) {




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
	for (int i = 0; i < next_num; i++)
		h_to_o[i].resize(next_num);


}
void hidden_layer::init() {
	

}