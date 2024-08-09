#include "MLP_Functions.h"
#include "rnn.h"
#include <cmath>
#include <windows.h>
// 2���� ���͸� �־��� �� ũ�⺰�� ������ �Լ�
std::vector<std::vector<std::vector<double>>> split2DVector(const std::vector<std::vector<double>>& vec, size_t chunkSize) {
	std::vector<std::vector<std::vector<double>>> result;

	for (size_t i = 0; i < vec.size(); i += chunkSize) {
		// 2���� ������ ���� �κ��� �����Ͽ� ���� 2���� ���͸� �����մϴ�.
		auto endIt = (i + chunkSize > vec.size()) ? vec.end() : vec.begin() + i + chunkSize;
		std::vector<std::vector<double>> chunk(vec.begin() + i, endIt);
		result.push_back(chunk);
	}

	return result;
}




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
		//y_train[i][0] = (y_train[i][0] - min_y) / (max_y - min_y);
	}
	int i = x_train[0].size();  // Number of input nodes
	int h = 30;  // Number of hidden nodes
	int o = 1;  // Number of output nodes

	RNN rnn(i, h, o, 0.01);
	double error = 0;

	int b_size = 100;
	vector<vector<double>> x_train_temp, y_train_temp;

	// ���͸� �ڸ� �� ũ�� ����
	size_t chunkSize = 100;
	// 2���� ���͸� �����ϴ�.
	std::vector<std::vector<std::vector<double>>> splitVecs_x = split2DVector(x_train, chunkSize);
	std::vector<std::vector<std::vector<double>>> splitVecs_y = split2DVector(y_train, chunkSize);
 
	for (int epoch = 0; epoch < 1000; epoch++) {


		for (int i = 0; i < splitVecs_x.size(); i++) {
			std::vector<std::vector<double>> sequenceOutputs = rnn.forward(splitVecs_x[i]);
			int j = 0;
			for (int t = splitVecs_x[i].size() * i; t < (i+1) *splitVecs_x[i].size(); t++) {
				error += (sequenceOutputs[j][0] - y_train[t][0]) * (sequenceOutputs[j][0] - y_train[t][0]);
				j++;
				//cout << "t is = " <<t  << " "<< endl;

				rnn.backward(splitVecs_x[i], splitVecs_y[i], 1);
			}
			error = error / (double)x_train.size();
			std::cout << "Epoch : " << epoch << " Error : " << error << std::endl;

			




			error = 0;
		}
		
	}

	return 0;
}