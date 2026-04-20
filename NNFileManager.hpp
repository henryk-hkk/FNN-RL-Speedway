#pragma once

#include "Dataset.hpp"
#include "FNN.hpp"
#include "LayerVectors.hpp"
#include <conio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <direct.h>
#include <ctime>
#include <iomanip>
#include <io.h>

class NNFileManager { //Neural Networks Files Manager
public:
    static Dataset getMNISTDatasetFromCSV(const std::string& filePath);

    static std::vector<double> getInputVectorFrom28x28BMP(const std::string& filePath);
    static void SaveMNISTDatasetToCSV(const Dataset& dataset, const std::string& filePath);
    static void saveFNN(const FNN& fnn, std::string savePath);
    static FNN loadFNN(std::string loadPath);
private:
    static void saveFNNLayerToCSV(const std::unique_ptr<FNNLayer>& layer, const std::string& filePath);
    static void saveFNNInputLayerToCSV(const std::unique_ptr<FNNLayer>& inputLayer, const std::string& filePath);
    static void saveFNNActivationFunctionsToCSV(const FNN& fnn, const std::string& savePath);
protected:
    static LayerVectors getFNNLayerVectors(const std::unique_ptr<FNNLayer>& layer);
    static LayerVectors getFNNLayerVectors(const std::string& filePath);

    static std::vector<AFunc> getActivationFunctions(const std::string& filePath);
    static std::tm getLocalTime();
    static size_t getFolderFileCount(const std::string& path);
};

