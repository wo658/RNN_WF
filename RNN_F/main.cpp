#include "MLP_Functions.h"
#include "rnn.h"
#include <cmath>
#include <windows.h>
// 2ì°¨ì› ë²¡í„°ë¥¼ ì£¼ì–´ì§„ í–‰ í¬ê¸°ë³„ë¡œ ë‚˜ëˆ„ëŠ” í•¨ìˆ˜
std::vector<std::vector<std::vector<double>>> split2DVector(const std::vector<std::vector<double>>& vec, size_t chunkSize) {
	std::vector<std::vector<std::vector<double>>> result;

	for (size_t i = 0; i < vec.size(); i += chunkSize) {
		// 2ì°¨ì› ë²¡í„°ì˜ í˜„ì¬ ë¶€ë¶„ì„ ì¶”ì¶œí•˜ì—¬ í•˜ìœ„ 2ì°¨ì› ë²¡í„°ë¥¼ ìƒì„±í•©ë‹ˆë‹¤.
		auto endIt = (i + chunkSize > vec.size()) ? vec.end() : vec.begin() + i + chunkSize;
		std::vector<std::vector<double>> chunk(vec.begin() + i, endIt);
		result.push_back(chunk);
	}

	return result;
}




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

	// ì •ê·œí™” ìµœëŒ€ ìµœì†Œ ë²”ìœ„ êµ¬í•˜ê¸°
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
	// ì •ê·œí™” 0 ~ 1 
	for (int i = 0; i < x_train.size(); i++) {
		for (int j = 0; j < x_train[0].size(); j++) {
			x_train[i][j] = (x_train[i][j] - min) / (max - min);
		}
<<<<<<< HEAD
		// ¼±ÅÃ»çÇ× 1 y Á¤±ÔÈ­
		// 1 . È¸±Í ¹®Á¦ÀÇ °æ¿ì Å¸°Ù°ªÀÌ ³ĞÀº ¹üÀ§¿¡ °ÉÃÄ ÀÖÀ» ¶§ Á¤±ÔÈ­
=======
		// ì„ íƒì‚¬í•­ 1 y ì •ê·œí™”
		// 1 . íšŒê·€ ë¬¸ì œì˜ ê²½ìš° íƒ€ê²Ÿê°’ì´ ë„“ì€ ë²”ìœ„ì— ê±¸ì³ ìˆì„ ë•Œ ì •ê·œí™”
>>>>>>> 6532bfa763ac38ce8fca9cd68361bb6f544ab0d3
		y_train[i][0] = (y_train[i][0] - min_y) / (max_y - min_y);
	}
	int i = x_train[0].size();  // Number of input nodes
	int h = 30;  // Number of hidden nodes
	int o = 1;  // Number of output nodes

	RNN rnn(i, h, o, 0.001);
	double error = 0;

	int b_size = 50;
	vector<vector<double>> x_train_temp, y_train_temp;

<<<<<<< HEAD
	// º¤ÅÍ¸¦ ÀÚ¸¦ Çà Å©±â ¼³Á¤
	size_t chunkSize = 50;
	// 2Â÷¿ø º¤ÅÍ¸¦ ³ª´¯´Ï´Ù.
=======
	// ë²¡í„°ë¥¼ ìë¥¼ í–‰ í¬ê¸° ì„¤ì •
	size_t chunkSize = 50;
	// 2ì°¨ì› ë²¡í„°ë¥¼ ë‚˜ëˆ•ë‹ˆë‹¤.
>>>>>>> 6532bfa763ac38ce8fca9cd68361bb6f544ab0d3
	std::vector<std::vector<std::vector<double>>> splitVecs_x = split2DVector(x_train, chunkSize);
	std::vector<std::vector<std::vector<double>>> splitVecs_y = split2DVector(y_train, chunkSize);
 
	for (int epoch = 0; epoch < 1000; epoch++) {

		
		for (int i = 0; i < splitVecs_x.size(); i++) {

			int j = 0;
<<<<<<< HEAD
				std::vector<std::vector<double>> sequenceOutputs = rnn.forward(splitVecs_x[i]);
			for (int t = splitVecs_x[i].size() * i; t < (i + 1) * splitVecs_x[i].size(); t++)
=======
			for (int t = splitVecs_x[i].size() * i; t < (i+1) *splitVecs_x[i].size(); t++) {
				std::vector<std::vector<double>> sequenceOutputs = rnn.forward(splitVecs_x[i]);
>>>>>>> 6532bfa763ac38ce8fca9cd68361bb6f544ab0d3
				error += (sequenceOutputs[j][0] - y_train[t][0]) * (sequenceOutputs[j][0] - y_train[t][0]);
				j++;
				//cout << "t is = " <<t  << " "<< endl;

				rnn.backward(splitVecs_x[i], splitVecs_y[i],sequenceOutputs,1);
<<<<<<< HEAD
				
			
			
=======
			}
			error = error / splitVecs_x.size();
			std::cout << "Epoch : " << epoch << " Error : " << error << std::endl;
>>>>>>> 6532bfa763ac38ce8fca9cd68361bb6f544ab0d3

			




		}
		error = error / x_train.size();
		std::cout << "Epoch : " << epoch << " Error : " << error << std::endl;

		error = 0;
		 /*
			int j = 0;
			
			std::vector<std::vector<double>> sequenceOutputs = rnn.forward(x_train);
			for (int t = 0; t <x_train.size(); t++) 
				error += (sequenceOutputs[j][0] - y_train[t][0]) * (sequenceOutputs[j][0] - y_train[t][0]);
			j++;
			//cout << "t is = " <<t  << " "<< endl;

			rnn.backward(x_train, y_train, sequenceOutputs, 1);
			
			error = error / x_train.size();
			std::cout << "Epoch : " << epoch << " Error : " << error << std::endl;






			error = 0;
		*/
	}

	return 0;
}
