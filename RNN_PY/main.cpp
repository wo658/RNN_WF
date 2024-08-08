#include "MLP_Functions.h"
#include "rnn.h"
#include <cmath>
#include <windows.h>
int main() {
	//Regression 
	//Data Load
	string dataPath = "C:\\Users\\whdgn\\Downloads\\RNN_WF-master\\RNN_WF-master\\RNN_PY\\ProcessDifference_train.csv";
	string dataPath_test = "C:\\Users\\whdgn\\Downloads\\RNN_WF-master\\RNN_WF-master\\RNN_PY\\ProcessDifference_test.csv";
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
	int i = x_train[0].size();  // Number of input nodes
	int h = 30;  // Number of hidden nodes
	int o = 1;  // Number of output nodes

	RNN rnn(i, h, o, 0.0005);
	double error = 0;
	for (int epoch = 0; epoch < 1000; ++epoch) {
		std::vector<std::vector<double>> sequenceOutputs = rnn.forward(x_train);
		rnn.backward(x_train, y_train);

		for (int t = 0; t < x_train.size(); t++)
			error += (sequenceOutputs[t][0] - y_train[t][0]) * (sequenceOutputs[t][0] - y_train[t][0]);
		
		if(epoch%10==0)
			std::cout << "Epoch " << epoch << ":" << error << std::endl;
		error = 0;
		
	}

	return 0;
}