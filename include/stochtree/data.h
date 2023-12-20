/*!
 * Copyright (c) 2023 randtree authors.
 * 
 * Classes for representing data in-memory and importing data from various formats (CSV file, 
 * R matrix, R dataframe, numpy array, pandas dataframe)
 * 
 * In supervised learning and causal inference problems, there are typically several types of variables
 *   - Covariates: often represented as X in statistics literature
 *   - Outcome: often represented as y in statistics literature
 *   - Treatment: often represented as Z in causal inference literature
 * 
 * Interface and class design inspired heavily by the Dataset and DatasetLoader 
 * classes in LightGBM, which is released under the following copyright:
 * 
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_DATA_H_
#define STOCHTREE_DATA_H_

#include <Eigen/Dense>

#include <stochtree/config.h>
#include <stochtree/io.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>

#include <vector>

namespace StochTree {

/*! \brief forward declaration of DataLoader class*/
class DataLoader;

/*! \brief Dataset used for training and predicting from stochastic tree models */
class Dataset {
 public:
  // Give DataLoader access to private members of Dataset
  friend DataLoader;

  /*! \brief Constructor */
  Dataset() {};

  /*! \brief Destructor */
  ~Dataset() {};

  /*! \brief Covariate value at a given row and col */
  double CovariateValue(data_size_t row, int32_t col) {return covariates_(row, col);}

  /*! \brief Treatment value at a given row and col */
  double TreatmentValue(data_size_t row, int32_t col) {return treatment_(row, col);}

  /*! \brief Outcome value at a given row and col */
  double OutcomeValue(data_size_t row, int32_t col) {return outcome_(row, col);}

  /*! \brief Residual value at a given row and col */
  double ResidualValue(data_size_t row, int32_t col) {return residuals_(row, col);}

  /*! \brief Outcome value at a given row (assuming a single outcome) */
  double OutcomeValue(data_size_t row) {return outcome_(row, 0);}

  /*! \brief Residual value at a given row (assuming a single outcome) */
  double ResidualValue(data_size_t row) {return residuals_(row, 0);}

  /*! \brief Add value to residual */
  inline void ResidualAdd(data_size_t row, int32_t col, double val) {
    residuals_(row, col) += val;
  }

  /*! \brief Subtract value from residual */
  inline void ResidualSubtract(data_size_t row, int32_t col, double val) {
    residuals_(row, col) -= val;
  }

  /*! \brief Divide residual by value */
  inline void ResidualDivide(data_size_t row, int32_t col, double val) {
    residuals_(row, col) /= val;
  }

  /*! \brief Divide residual by value */
  inline void ResidualMultiply(data_size_t row, int32_t col, double val) {
    residuals_(row, col) *= val;
  }

  /*! \brief Reset all residuals to raw outcome values */
  inline void ResidualReset() {
    data_size_t const resid_rows = residuals_.rows();
    int const resid_cols = residuals_.cols();
    data_size_t const outcome_rows = outcome_.rows();
    int const outcome_cols = outcome_.cols();

    if ((resid_rows != outcome_rows) || (resid_cols != outcome_cols)) {
      Log::Fatal("Residual matrix is a different size than outcome matrix");
    }
    for (data_size_t i = 0; i < outcome_rows; i++) {
      for (int j = 0; j < outcome_cols; j++) {
        residuals_(i, j) = outcome_(i, j);
      }
    }
  }

  /*! \brief Number of observations in the dataset */
  int32_t NumObservations() {return num_observations_;}

  /*! \brief Whether or not a dataset has covariates loaded */
  bool HasCovariates() {return num_covariates_ > 0;}

  /*! \brief Whether or not a dataset has treatment variable loaded */
  bool HasTreatment() {return num_treatment_ > 0;}

  /*! \brief Whether or not a dataset has outcome variable loaded */
  bool HasOutcome() {return num_outcome_ > 0;}

  /*! \brief Number of outcome variables */
  int32_t NumOutcome() {return num_outcome_;}

  /*! \brief Number of treatment variables */
  int32_t NumTreatment() {return num_treatment_;}

  /*! \brief Number of basis variables */
  int32_t NumBasis() {return num_basis_;}

  /*! \brief Number of Numeric Covariates */
  int32_t NumCovariates() {return num_covariates_;}

  /*! \brief Number of Numeric Covariates */
  int32_t NumNumericCovariates() {return num_numeric_covariates_;}

  /*! \brief Number of Ordered Categorical Covariates */
  int32_t NumOrderedCategoricalCovariates() {return num_ordered_categorical_covariates_;}

  /*! \brief Number of Unordered Categorical Covariates */
  int32_t NumUnorderedCategoricalCovariates() {return num_unordered_categorical_covariates_;}

  /*! \brief Whether outcome is univariate */
  bool UnivariateOutcome() {return univariate_outcome_;}

  /*! \brief Whether treatment is univariate */
  bool UnivariateTreatment() {return univariate_treatment_;}

  /*! \brief Whether basis is univariate */
  bool UnivariateBasis() {return univariate_basis_;}

  /*! \brief Type of feature j */
  FeatureType GetFeatureType(int32_t j) {return covariate_types_[j];}
  
 private:
  // Raw data, stored in Eigen matrices
  Eigen::MatrixXd covariates_;
  Eigen::MatrixXd treatment_;
  Eigen::MatrixXd outcome_;
  Eigen::MatrixXd residuals_;
  Eigen::MatrixXd basis_;
  data_size_t num_observations_;

  // Covariate info
  int32_t num_covariates_{0};
  int32_t num_numeric_covariates_{0};
  int32_t num_ordered_categorical_covariates_{0};
  int32_t num_unordered_categorical_covariates_{0};
  std::vector<FeatureType> covariate_types_;

  // Outcome, treatment, and basis function dimensionality info
  int32_t num_outcome_{0};
  int32_t num_treatment_{0};
  int32_t num_basis_{0};
  bool univariate_outcome_{true};
  bool univariate_treatment_{true};
  bool univariate_basis_{true};
};

/*! \brief Dataset creation class. Can build a training dataset by either:
 *     (1) Parsing CSV files (no other file types supported at present)
 *     (2) Reading contiguous-memory data from an R matrix or numpy array
 */
class DataLoader {
 public:
  DataLoader(const Config& io_config, int num_class, const char* filename);

  ~DataLoader();

  Dataset* LoadFromFile(const char* filename);

  Dataset* ConstructFromMatrix(double* matrix_data, int num_col, 
                               data_size_t num_row, bool is_row_major);

  /*! \brief Disable copy */
  DataLoader& operator=(const DataLoader&) = delete;
  DataLoader(const DataLoader&) = delete;

 private:
  void LoadHeaderFromMemory(Dataset* dataset, const char* buffer);

  void SetHeader(const Config& io_config);

  void UnpackColumnVectors(std::vector<int32_t>& label_columns, std::vector<int32_t>& outcome_columns, 
                           std::vector<int32_t>& ordered_categoricals, std::vector<int32_t>& unordered_categoricals, 
                           std::vector<int32_t>& basis_columns);

  void CheckDataset(const Dataset* dataset);

  std::vector<std::string> LoadTextDataToMemory(const char* filename, int* num_global_data);

  /*! \brief Extract local features from memory */
  void ExtractFeaturesFromMemory(std::vector<std::string>* text_data, const Parser* parser, Dataset* dataset);
  
  /*! \brief Config object used to drive prediction*/
  const Config& config_;
  /*! \brief store feature names */
  std::vector<std::string> variable_names_;
  /*! \brief indices of outcomes in the data */
  std::vector<int32_t> outcome_columns_;
  /*! \brief indices of treatment variable in the data */
  std::vector<int32_t> treatment_columns_;
  /*! \brief indices of unordered categorical features */
  std::vector<int32_t> unordered_categoricals_;
  /*! \brief indices of ordered categorical features */
  std::vector<int32_t> ordered_categoricals_;
  /*! \brief indices of basis functions */
  std::vector<int32_t> basis_columns_;
};

} // namespace StochTree

#endif   // STOCHTREE_DATA_H_
