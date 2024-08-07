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

	// ����ȭ �ִ� �ּ� ���� ���ϱ�
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
	// ����ȭ 0 ~ 1 
	for (int i = 0; i < x_train.size(); i++) {
		for (int j = 0; j < x_train[0].size(); j++) {
			x_train[i][j] = (x_train[i][j] - min) / (max - min);
		}
		// ���û��� 1 y ����ȭ
		// 1 . ȸ�� ������ ��� Ÿ�ٰ��� ���� ������ ���� ���� �� ����ȭ
		y_train[i][0] = (y_train[i][0] - min_y) / (max_y - min_y);
	}




	hidden_layer h(x_train[0].size(), 30 , 1);
	in_layer i(x_train[0].size());
	out_layer o(1);

	h.init();


	rnn rnn(h,o,i);
	double error = 0;
	for (int epoch = 1; epoch < 100; epoch++) {
	
	//	rnn.feedforward();
	//	rnn.backpropagation();
	//	error = rnn.loss_cal();

		cout <<epoch << "epoch is : " << error << endl;
	}

	system("pause");
}

