/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <stochtree/io.h>
#include <stochtree/data.h>

#include <chrono>
#include <cstdio>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace StochTree {

DataLoader::DataLoader(const Config& io_config, int num_class, const char* filename)
  :config_(io_config) {
  UnpackColumnVectors(outcome_columns_, treatment_columns_, ordered_categoricals_, unordered_categoricals_, basis_columns_);
}

DataLoader::~DataLoader() {
}

void DataLoader::UnpackColumnVectors(std::vector<int32_t>& outcome_columns, std::vector<int32_t>& treatment_columns, 
                                     std::vector<int32_t>& ordered_categoricals, std::vector<int32_t>& unordered_categoricals, 
                                     std::vector<int32_t>& basis_columns) {
  // Read label indices from config
  if (config_.outcome_columns.size() > 0) {
    outcome_columns = config_.Str2FeatureVec(config_.outcome_columns.c_str());
  }
  
  // Read treatment indices from config
  if (config_.treatment_columns.size() > 0) {
    treatment_columns = config_.Str2FeatureVec(config_.treatment_columns.c_str());
  }
  
  // Read ordered categorical variables from config
  if (config_.ordered_categoricals.size() > 0) {
    ordered_categoricals = config_.Str2FeatureVec(config_.ordered_categoricals.c_str());
  }

  // Read unordered categorical variables from config
  if (config_.unordered_categoricals.size() > 0) {
    unordered_categoricals = config_.Str2FeatureVec(config_.unordered_categoricals.c_str());
  }

  // Read basis columns from config
  if (config_.basis_columns.size() > 0) {
    basis_columns = config_.Str2FeatureVec(config_.basis_columns.c_str());
  }
}


Dataset* DataLoader::LoadFromFile(const char* filename) {
  auto dataset = std::unique_ptr<Dataset>(new Dataset());
  data_size_t num_global_data = 0;
  auto parser = std::unique_ptr<Parser>(Parser::CreateParser(filename, config_.header, 0, config_.precise_float_parser));
  if (parser == nullptr) {
    Log::Fatal("Could not recognize data format of %s", filename);
  }
  
  // Determine number of columns in the data file
  int num_columns = parser->NumFeatures();

  // Read data to memory
  auto text_data = LoadTextDataToMemory(filename, &num_global_data);
  dataset->num_observations_ = static_cast<data_size_t>(text_data.size());

  // Check if there are outcome columns
  int32_t num_outcome = outcome_columns_.size();
  bool has_outcome = num_outcome > 0;
  // Resize outcome vector
  if (has_outcome) {
    if (num_outcome == 1) dataset->univariate_outcome_ = true;
    else dataset->univariate_outcome_ = false;
    dataset->outcome_.resize(dataset->num_observations_, num_outcome);
    dataset->residuals_.resize(dataset->num_observations_, num_outcome);
    dataset->num_outcome_ = num_outcome;
  }

  // Check if there are treatment columns
  int32_t num_treatment = treatment_columns_.size();
  bool has_treatment = num_treatment > 0;
  // Resize treatment vector
  if (has_treatment) {
    if (num_treatment == 1) dataset->univariate_treatment_ = true;
    else dataset->univariate_treatment_ = false;
    dataset->treatment_.resize(dataset->num_observations_, num_treatment);
    dataset->num_treatment_ = num_treatment;
  }

  // Check if there are basis columns
  int32_t num_basis = basis_columns_.size();
  bool has_basis = num_basis > 0;
  // Resize treatment vector
  if (has_basis) {
    if (num_basis == 1) dataset->univariate_basis_ = true;
    else dataset->univariate_basis_ = false;
    dataset->basis_.resize(dataset->num_observations_, num_basis);
    dataset->num_basis_ = num_basis;
  }
  
  // Determine the number of features (as opposed to outcome / treatment / basis function)
  int num_features = 0;
  bool outcome_matched, treatment_matched, basis_matched;
  for (int i = 0; i < num_columns; i++){
    outcome_matched = (std::find(outcome_columns_.begin(), outcome_columns_.end(), i)
                        != outcome_columns_.end());
    treatment_matched = (std::find(treatment_columns_.begin(), treatment_columns_.end(), i)
                        != treatment_columns_.end());
    basis_matched = (std::find(basis_columns_.begin(), basis_columns_.end(), i)
                     != basis_columns_.end());
    if (!outcome_matched && !treatment_matched && !basis_matched) {
      num_features += 1;
    }
  }
  Log::Info("Num features = %d", num_features);

  // Resize covariate vector
  if (num_features > 0) {
    dataset->covariates_.resize(dataset->num_observations_, num_features);
  }
  dataset->num_covariates_ = num_features;

  // Unpack covariate types
  bool ordered_categorical_matched;
  bool unordered_categorical_matched;
  for (int j = 0; j < num_features; j++) {
    ordered_categorical_matched = (std::find(ordered_categoricals_.begin(), ordered_categoricals_.end(), j)
                                   != ordered_categoricals_.end());
    unordered_categorical_matched = (std::find(unordered_categoricals_.begin(), unordered_categoricals_.end(), j)
                                     != unordered_categoricals_.end());
    if (ordered_categorical_matched) {
      dataset->covariate_types_.push_back(FeatureType::kOrderedCategorical);
      dataset->num_ordered_categorical_covariates_++;
    } else if (unordered_categorical_matched) {
      dataset->covariate_types_.push_back(FeatureType::kUnorderedCategorical);
      dataset->num_unordered_categorical_covariates_++;
    } else {
      dataset->covariate_types_.push_back(FeatureType::kNumeric);
      dataset->num_numeric_covariates_++;
    }
  }
  
  // Unpack data from file
  ExtractFeaturesFromMemory(&text_data, parser.get(), dataset.get());
  text_data.clear();
  
  // Make sure an in-memory dataset was successfully created
  CheckDataset(dataset.get());

  // Release the in memory dataset pointer
  return dataset.release();
}

Dataset* DataLoader::ConstructFromMatrix(double* matrix_data, int num_col, data_size_t num_row, bool is_row_major){
  auto dataset = std::unique_ptr<Dataset>(new Dataset());
  dataset->num_observations_ = num_row;
  
  // Check if there are outcome columns
  int32_t num_outcome = outcome_columns_.size();
  bool has_outcome = num_outcome > 0;
  if (has_outcome) {
    if (num_outcome == 1) dataset->univariate_outcome_ = true;
    else dataset->univariate_outcome_ = false;
    dataset->outcome_.resize(dataset->num_observations_, num_outcome);
    dataset->residuals_.resize(dataset->num_observations_, num_outcome);
    dataset->num_outcome_ = num_outcome;
  }

  // Check if there are treatment columns
  int32_t num_treatment = treatment_columns_.size();
  bool has_treatment = num_treatment > 0;
  if (has_treatment) {
    if (num_treatment == 1) dataset->univariate_treatment_ = true;
    else dataset->univariate_treatment_ = false;
    dataset->treatment_.resize(dataset->num_observations_, num_treatment);
    dataset->num_treatment_ = num_treatment;
  }

  // Check if there are basis columns
  int32_t num_basis = basis_columns_.size();
  bool has_basis = num_basis > 0;
  // Resize treatment vector
  if (has_basis) {
    if (num_basis == 1) dataset->univariate_basis_ = true;
    else dataset->univariate_basis_ = false;
    dataset->basis_.resize(dataset->num_observations_, num_basis);
    dataset->num_basis_ = num_basis;
  }
  
  // Determine the number of features (as opposed to outcome / treatment)
  int num_features = 0;
  bool outcome_matched, treatment_matched, basis_matched;
  for (int i = 0; i < num_col; i++){
    outcome_matched = (std::find(outcome_columns_.begin(), outcome_columns_.end(), i)
                        != outcome_columns_.end());
    treatment_matched = (std::find(treatment_columns_.begin(), treatment_columns_.end(), i)
                        != treatment_columns_.end());
    basis_matched = (std::find(basis_columns_.begin(), basis_columns_.end(), i)
                     != basis_columns_.end());
    if (!outcome_matched && !treatment_matched && !basis_matched) {
      num_features += 1;
    }
  }
  Log::Info("Num features = %d", num_features);
  if (num_features > 0) {
    dataset->covariates_.resize(dataset->num_observations_, num_features);
  }
  dataset->num_covariates_ = num_features;

  // Unpack covariate types
  bool ordered_categorical_matched;
  bool unordered_categorical_matched;
  for (int j = 0; j < num_features; j++) {
    ordered_categorical_matched = (std::find(ordered_categoricals_.begin(), ordered_categoricals_.end(), j)
                                   != ordered_categoricals_.end());
    unordered_categorical_matched = (std::find(unordered_categoricals_.begin(), unordered_categoricals_.end(), j)
                                     != unordered_categoricals_.end());
    if (ordered_categorical_matched) {
      dataset->covariate_types_.push_back(FeatureType::kOrderedCategorical);
      dataset->num_ordered_categorical_covariates_++;
    } else if (unordered_categorical_matched) {
      dataset->covariate_types_.push_back(FeatureType::kUnorderedCategorical);
      dataset->num_unordered_categorical_covariates_++;
    } else {
      dataset->covariate_types_.push_back(FeatureType::kNumeric);
      dataset->num_numeric_covariates_++;
    }
  }

  double temp_value;
  int feature_counter, outcome_counter, treatment_counter, basis_counter;
  for (data_size_t i = 0; i < dataset->num_observations_; ++i) {
    feature_counter = 0;
    outcome_counter = 0;
    treatment_counter = 0;
    basis_counter = 0;
    for (int j = 0; j < num_col; ++j) {
      if (is_row_major){
        // Numpy 2-d arrays are stored in "row major" order
        temp_value = static_cast<double>(*(matrix_data + static_cast<data_size_t>(num_col) * i + j));
      } else {
        // R matrices are stored in "column major" order
        temp_value = static_cast<double>(*(matrix_data + static_cast<data_size_t>(num_row) * j + i));
      }
      
      // Unpack data into outcome, treatment or covariates
      outcome_matched = (std::find(outcome_columns_.begin(), outcome_columns_.end(), j)
                          != outcome_columns_.end());
      treatment_matched = (std::find(treatment_columns_.begin(), treatment_columns_.end(), j)
                          != treatment_columns_.end());
      basis_matched = (std::find(basis_columns_.begin(), basis_columns_.end(), i)
                       != basis_columns_.end());
      if (outcome_matched){
        dataset->outcome_(i, outcome_counter) = temp_value;
        dataset->residuals_(i + outcome_counter) = temp_value;
        outcome_counter += 1;
      } else if (treatment_matched) {
        dataset->treatment_(i, treatment_counter) = temp_value;
        treatment_counter += 1;
      } else if (basis_matched) {
        dataset->basis_(i, basis_counter) = temp_value;
        basis_counter += 1;
      } else {
        dataset->covariates_(i, feature_counter) = temp_value;
        feature_counter += 1;
      }
    }
  }
  
  // Make sure an in-memory dataset was successfully created
  CheckDataset(dataset.get());

  // Release the in memory dataset pointer
  return dataset.release();
}

void DataLoader::CheckDataset(const Dataset* dataset) {
  if (dataset->num_observations_ <= 0) {
    Log::Fatal("Data loaded was empty");
  }
}

std::vector<std::string> DataLoader::LoadTextDataToMemory(const char* filename, int* num_global_data) {
  TextReader<data_size_t> text_reader(filename, config_.header, config_.file_load_progress_interval_bytes);
  // read all lines
  *num_global_data = text_reader.ReadAllLines();
  return std::move(text_reader.Lines());
}

/*! \brief Extract local features from memory */
void DataLoader::ExtractFeaturesFromMemory(std::vector<std::string>* text_data, const Parser* parser, Dataset* dataset) {
  std::vector<std::pair<int, double>> oneline_features;
  auto& ref_text_data = *text_data;
  int feature_counter, outcome_counter, treatment_counter, basis_counter;
  bool outcome_matched, treatment_matched, basis_matched;
  for (data_size_t i = 0; i < dataset->num_observations_; ++i) {
    // unpack the vector of textlines read from file into a vector of (int, double) tuples
    oneline_features.clear();
    parser->ParseOneLine(ref_text_data[i].c_str(), &oneline_features);
    
    // free processed line:
    ref_text_data[i].clear();

    // unload the data from oneline_features vector into the dataset variables containers    
    feature_counter = 0;
    outcome_counter = 0;
    treatment_counter = 0;
    basis_counter = 0;
    for (auto& inner_data : oneline_features) {
      int feature_idx = inner_data.first;
      outcome_matched = (std::find(outcome_columns_.begin(), outcome_columns_.end(), feature_idx)
                          != outcome_columns_.end());
      treatment_matched = (std::find(treatment_columns_.begin(), treatment_columns_.end(), feature_idx)
                          != treatment_columns_.end());
      basis_matched = (std::find(basis_columns_.begin(), basis_columns_.end(), feature_idx)
                       != basis_columns_.end());
      if (outcome_matched){
        dataset->outcome_(i, outcome_counter) = inner_data.second;
        dataset->residuals_(i, outcome_counter) = inner_data.second;
        outcome_counter += 1;
      } else if (treatment_matched) {
        dataset->treatment_(i, treatment_counter) = inner_data.second;
        treatment_counter += 1;
      } else if (basis_matched) {
        dataset->basis_(i, basis_counter) = inner_data.second;
        basis_counter += 1;
      } else {
        dataset->covariates_(i, feature_counter) = inner_data.second;
        feature_counter += 1;
      }
    }
  }
  // free text data after use
  text_data->clear();
}

}  // namespace StochTree
