// Inclure les bibliothèques nécessaires
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>

//Path
const char* MODEL_PATH = "loc.tflite";
const char* FINGERPRINT_DATA_PATH = "empreintes_digitales.txt";

// harger le modele .tflite
std::unique_ptr<tflite::Interpreter> LoadModel(const char* model_path) {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_path);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  interpreter->AllocateTensors();
  return interpreter;
}

// charger les données d'empreintes digitales  text file
std::vector<float> LoadFingerprintData(const char* data_path) {
  std::vector<float> fingerprint_data;
  std::ifstream data_file(data_path);
  float value;
  while (data_file >> value) {
    fingerprint_data.push_back(value);
  }
  return fingerprint_data;
}

// effectuer la correspondance d'empreintes digitales
float PerformLatentFingerprintMatching(const std::vector<float>& fingerprint_data, tflite::Interpreter* interpreter) {
  // Obtenir les informations sur les tenseurs d'entrée et de sortie du modèle
  int input_tensor_index = interpreter->inputs()[0];
  TfLiteIntArray* input_dims = interpreter->tensor(input_tensor_index)->dims;
  int num_fingerprints = input_dims->data[0];
  int fingerprint_size = input_dims->data[1];

  int output_tensor_index = interpreter->outputs()[0];
  TfLiteIntArray* output_dims = interpreter->tensor(output_tensor_index)->dims;
  int num_classes = output_dims->data[1];

  // Vérifier si les dimensions
  if (fingerprint_data.size() != num_fingerprints * fingerprint_size) {
    std::cerr << "Erreur: Les dimensions des données d'empreintes digitales ne correspondent pas aux dimensions du modèle." << std::endl;
    return -1.0;
  }

  //tensor
  float* input_data = interpreter->typed_tensor<float>(input_tensor_index);
  for (int i = 0; i < fingerprint_data.size(); i++) {
    input_data[i] = fingerprint_data[i];
  }

  // Exécut
  interpreter->Invoke();

  // Obtenir les résultats de la correspondance d'empreintes digitales
  float* output_data = interpreter->typed_tensor<float>(output_tensor_index);
  float match_score = output_data[0];

  return match_score;
}

int main() {
  // Charger le modèle CNN
  std::unique_ptr<tflite::Interpreter> interpreter = LoadModel(MODEL_PATH);

  // Charger les données d'empreintes digitales
  std::vector<float> fingerprint_data = LoadFingerprintData(FINGERPRINT_DATA_PATH);

  // correspondance d'empreintes
  float match_score = PerformLatentFingerprintMatching(fingerprint_data, interpreter.get());

  // Afficher le score de correspondance
  std::cout << "Score de correspondance: " << match_score << std::endl;

  return 0;
}