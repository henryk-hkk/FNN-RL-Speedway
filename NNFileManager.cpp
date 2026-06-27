#include "NNFileManager.hpp"

Dataset NNFileManager::getMNISTDatasetFromCSV(const std::string& filePath) {
    Dataset dataset;
    std::ifstream file(filePath);
    if (!file.good()) {
        std::cout << "Provided CSV file is incorrect.\n";
        return Dataset{ {}, {} };
    }

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        unsigned i{};
        while (std::getline(ss, cell, ',')) {
            if (i == 0) {
                dataset.desiredOutputVector.push_back(mfuncs::convertDigitToOutputVector(std::stoi(cell)));
                i++;
                continue;
            }
            row.push_back(std::stod(cell) / 255.0);
        }
        dataset.inputVector.push_back(row);
    }
    file.close();

    return dataset;
}

std::vector<double> NNFileManager::getInputVectorFrom28x28BMP(const std::string& filePath) {
    std::vector<double> inputVec;
    inputVec.reserve(28 * 28);
    std::ifstream file;

    file.open(filePath, std::ios::in | std::ios::binary);

    uint32_t dataOffset{};
    file.seekg(10);
    file.read(reinterpret_cast<char*>(&dataOffset), 4);
    file.seekg(dataOffset);
    //file header + information

    int rowSize = ((24 * 28 + 31) / 32) * 4;
    int padding = rowSize - 28 * 3;

    for (int y{ 27 }; y >= 0; y--) {
        for (int x{}; x < 28; x++) {
            unsigned char r, g, b;
            file.read(reinterpret_cast<char*>(&b), 1);
            file.read(reinterpret_cast<char*>(&g), 1);
            file.read(reinterpret_cast<char*>(&r), 1);

            double gray = (r + g + b) / 3.0;

            double mnistPixel = gray;

            inputVec.push_back(mnistPixel);
        }
        file.ignore(padding);
    }
    return inputVec;
}
void NNFileManager::SaveMNISTDatasetToCSV(const Dataset & dataset, const std::string & filePath) {
    std::ofstream file(filePath);
    for (size_t i{}; i < dataset.inputVector.size(); i++) {
        file << mfuncs::convertOutputVectorToDigit(dataset.desiredOutputVector[i]) << ",";
        for (size_t inputPixel{}; inputPixel < dataset.inputVector[i].size(); inputPixel++) {
            file << dataset.inputVector[i][inputPixel] << ",";
        }
        file << "\n";
    }
    file.close();
}
void NNFileManager::saveFNN(const FNN& fnn, std::string savePath) {
    std::locale::global(std::locale("C"));
    std::tm time = getLocalTime();
    size_t layerCount = fnn.layers.size();
    std::ostringstream oss;
    oss << "/neural_network_" << std::put_time(&time, "%Y-%m-%d_%H-%M-%S");
    savePath += oss.str();

    int status = _mkdir(savePath.c_str());
    if (status != 0) {
        std::cout << "saving failed!";
        return;
    }
    saveFNNInputLayerToCSV(fnn.layers[0], savePath + "/layer0");
    for (size_t i{ 1 }; i < layerCount; i++) {
        saveFNNLayerToCSV(fnn.layers[i], savePath + "/layer" + std::to_string(i));
    }
    saveFNNActivationFunctionsToCSV(fnn, savePath);
}
FNN NNFileManager::loadFNN(std::string loadPath) {
    FNN fnn;
    int fileCount = getFolderFileCount(loadPath);

    std::ifstream inputLayer(loadPath + "/layer0_input.csv");
    std::string inputLayerSizeStr;
    std::getline(inputLayer, inputLayerSizeStr);
    size_t inputLayerSize = stoi(inputLayerSizeStr);
    fnn.addInputLayer(inputLayerSize);
    inputLayer.close();

    fileCount -= 2;//ommiting the input layer file and activation functions files
    size_t layerCount = (fileCount / 2) + 1;

    auto funcs = getActivationFunctions(loadPath + "/functions.csv");
    fnn.setLayerActivationFunction(0, funcs[0]);
    for (size_t i{ 1 }; i < layerCount; i++) {
        auto lV = getFNNLayerVectors(loadPath + "/layer" + std::to_string(i));
        size_t neuronCount = lV.biases.size();
        if (i == layerCount - 1) fnn.addOutputLayer(neuronCount);
        else fnn.addLayer(neuronCount);
        fnn.layers[i]->weights = lV.weights;
        for (size_t j{}; j < neuronCount; j++)
            fnn.layers[i]->neurons[j]->bias = lV.biases[j];
        fnn.setLayerActivationFunction(i, funcs[i]);
    }

    return fnn;
}

void NNFileManager::saveFNNLayerToCSV(const std::unique_ptr<FNNLayer>& layer, const std::string& filePath) {
    LayerVectors lV = getFNNLayerVectors(layer);
    std::ofstream weights(filePath + "_weights.csv"), biases(filePath + "_biases.csv");
    bool firstLineFlag = true;
    for (size_t neuron{}; neuron < lV.weights.size(); neuron++) {
        biases << lV.biases[neuron] << ",";
        if (firstLineFlag) firstLineFlag = false;
        else weights << "\n";
        for (size_t previousNeuron{}; previousNeuron < lV.weights[neuron].size(); previousNeuron++)
            weights << lV.weights[neuron][previousNeuron] << ",";
    }
    weights.close();
    biases.close();
}
void NNFileManager::saveFNNInputLayerToCSV(const std::unique_ptr<FNNLayer>& inputLayer, const std::string& filePath) {
    std::ofstream input(filePath + "_input.csv");
    input << inputLayer->neuronCount;
    input.close();
    return;
}
void NNFileManager::saveFNNActivationFunctionsToCSV(const FNN& fnn, const std::string& savePath) {
    std::ofstream functions(savePath + "/functions.csv");
    for (size_t layer{}; layer < fnn.layers.size(); layer++) {
        functions << static_cast<int>(fnn.layers[layer]->activationFunc) << ",";
    }
    functions.close();
}

LayerVectors NNFileManager::getFNNLayerVectors(const std::unique_ptr<FNNLayer>& layer) {
    LayerVectors lV;
    lV.weights = layer->weights;
    for (size_t neuron{}; neuron < layer->neuronCount; neuron++)
        lV.biases.push_back(layer->neurons[neuron]->bias);
    return lV;
}
LayerVectors NNFileManager::getFNNLayerVectors(const std::string& filePath) {
    std::string line, cell;

    std::vector<double> biases;
    std::vector<std::vector<double>> weights;

    std::ifstream layerBiasFile(filePath + "_biases.csv");
    std::getline(layerBiasFile, line);
    std::stringstream ss(line);

    while (std::getline(ss, cell, ','))
        if(!cell.empty()) biases.push_back(std::stod(cell));

    layerBiasFile.close();

    std::vector<double> row;
    std::ifstream layerWeightsFile(filePath + "_weights.csv");
    while (std::getline(layerWeightsFile, line)) {
        std::stringstream ss(line);
        while (std::getline(ss, cell, ','))
            if (!cell.empty()) row.push_back(std::stod(cell));
        weights.push_back(row);
        row.clear();
    }
    layerWeightsFile.close();

    LayerVectors lV{ weights, biases };
    return lV;
}

std::vector<AFunc> NNFileManager::getActivationFunctions(const std::string& filePath) {
    std::ifstream activationFunctionsFile(filePath);
    std::string line, cell;
    std::vector<AFunc> activationFunctions;
    while (std::getline(activationFunctionsFile, line)) { //file should only contain one line anyway
        std::stringstream ss(line);
        while (std::getline(ss, cell, ','))
            activationFunctions.push_back(static_cast<AFunc>(std::stoi(cell)));
    }
    return activationFunctions;
}

std::tm NNFileManager::getLocalTime() {
    std::time_t t = std::time(nullptr);
    std::tm time{};
    localtime_s(&time, &t);
    return time;
}
size_t NNFileManager::getFolderFileCount(const std::string& path) {
    _finddata_t data;
    intptr_t handle = _findfirst((path + "\\*").c_str(), &data);
    int count = (handle != -1);

    while (_findnext(handle, &data) == 0)
        ++count;

    _findclose(handle);
    return count - 2;
}
