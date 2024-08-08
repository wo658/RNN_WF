#include "MLP_Functions.h"
#include "Layer.h"
#include "rnn.h"
#include <cmath>
#include <windows.h>
int main() {
	//Regression 
	//Data Load
	string dataPath = "C:\\Users\\ecmdev\\Desktop\\123\\MLP_Ex\\MLP_Ex\\ProcessDifference_train.csv";
	string dataPath_test = "C:\\Users\\ecmdev\\Desktop\\123\\MLP_Ex\\MLP_Ex\\ProcessDifference_test.csv";
	const char* NameofData = dataPath.c_str();
	const char* testData = dataPath.c_str();
	vector<vector<double>> train;
	vector<vector<double>> test;
	train = readFile(NameofData);
	test = readFile(testData);


	//x_train/Y_train Split
	vector<vector<double>> x_train, y_train;
	vector<vector<double>> x_test, y_test;
	splitData(train, x_train, y_train);
	splitData(test, x_test, y_test);

	// 정규화 최대 최소 범위 구하기
	double max = x_train[0][0], min = x_train[0][0];
	double max_y = y_train[0][0], min_y = y_train[0][0];
	for (int i = 0; i < x_train.size(); i++) {
		for (int j = 0; j < x_train[0].size(); j++) {
			if (max < x_train[i][j])
				max = x_train[i][j];
			if (min > x_train[i][j])
				min = x_train[i][j];
		}
		if (max_y < y_train[i][0])
			max_y = y_train[i][0];
		if (min_y > y_train[i][0])
			min_y = y_train[i][0];
	}

	//cout << max << " " << min << endl;
	// 정규화 0 ~ 1 
	for (int i = 0; i < x_train.size(); i++) {
		for (int j = 0; j < x_train[0].size(); j++) {
			x_train[i][j] = (x_train[i][j] - min) / (max - min);
		}
		// 선택사항 1 y 정규화
		// 1 . 회귀 문제의 경우 타겟값이 넓은 범위에 걸쳐 있을 때 정규화
		y_train[i][0] = (y_train[i][0] - min_y) / (max_y - min_y);
	}



	// 초기화
	hidden_layer h(x_train[0].size(),30 , 1);
	in_layer i(x_train[0].size());
	out_layer o(1);
	h.init();
	rnn rnn(h,o,i,x_train.size());
	double output = 0;
	double exp,error=0;



	for (int epoch = 1; epoch < 10; epoch++) {
		for (int train_size = 0; train_size < x_train.size(); train_size++) {

			rnn.feedforward(x_train[train_size],train_size);
			
			error = (rnn.outputs[train_size] - y_train[train_size][0]) * (rnn.outputs[train_size] - y_train[train_size][0]);

			if(train_size%1000 ==0)
				cout << train_size << "error is : " << error << endl;



		}
		
		rnn.backpropagation(x_train,y_train);
		
		error = 0;

	}

	system("pause");
}

