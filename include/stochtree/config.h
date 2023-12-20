/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef STOCHTREE_CONFIG_H_
#define STOCHTREE_CONFIG_H_

#include <stochtree/common.h>
#include <stochtree/export.h>
#include <stochtree/log.h>
#include <stochtree/meta.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace StochTree {

/*! \brief Types of tasks */
enum TaskType {
  kSupervisedLearning, kBinaryTreatmentEffect, kContinuousTreatmentEffect
};

/*! \brief Types of outcomes */
enum OutcomeType {
  kContinuous, kBinary, KOrdinal
};

/*! \brief Types of sampling methods */
enum MethodType {
  kXBART, kBART, kWarmStartBART
};

struct Config {
 public:
  std::string ToString() const;
  /*!
  * \brief Get string value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetString(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, std::string* out);

  /*!
  * \brief Get int value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetInt(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, int* out);

  /*!
  * \brief Get double value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetDouble(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, double* out);

  /*!
  * \brief Get bool value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetBool(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, bool* out);

  /*!
   * \brief Sort aliases by length and then alphabetically
   * \param x Alias 1
   * \param y Alias 2
   * \return true if x has higher priority than y
   */
  inline static bool SortAlias(const std::string& x, const std::string& y);
  
  /*!
   * \brief Deduplicate an input config by retaining only the first observed instance of each parameter
   * \param params Initial key-value pair of parameter names and values
   * \param out Deduplicated key-value pair of parameter names and values
   */
  static void KeepFirstValues(const std::unordered_map<std::string, std::vector<std::string>>& params, std::unordered_map<std::string, std::string>* out);
  
  /*!
   * \brief Set the verbosity of the program
   * \param params Key value pair of parameter names and values
   */
  static void SetVerbosity(const std::unordered_map<std::string, std::vector<std::string>>& params);

  /*!
   * \brief Convert a string of the form "param_name=param_value, ..." to a key-value pair
   * \param parameters Pointer to character array of form "param_name=param_value, ..."
   */
  static std::unordered_map<std::string, std::string> Str2Map(const char* parameters);
  
  /*!
   * \brief Helper function for Str2Map
   * \param params Key value pair to be populated by parsed string
   * \param kv Pointer to character array of form "param_name=param_value, ..."
   */
  static void KV2Map(std::unordered_map<std::string, std::vector<std::string>>* params, const char* kv);

  /*!
   * \brief Convert a string of the form "0,2,3 ..." to a vector of feature indices
   * \param parameters Pointer to character array of form "0,2,3 ..."
   */
  static std::vector<int> Str2FeatureVec(const char* parameters);

  /*!
   * \brief Helper function for Str2FeatureVec
   * \param categorical_variables Vector of covariate indices to be parsed
   * \param kv Pointer to character array of form "0,2,3..."
   */
  static void FeatureUnpack(std::vector<int>* categorical_variables, const char* var_id);

  // [no-save]
  // [doc-only]
  // type = enum
  // default = supervised_learning
  // options = supervised_learning, binary_treatment_effect, continuous_treatment_effect
  // desc = ``supervised_learning``, for prediction of outcome given a set of covariates
  // desc = ``binary_treatment_effect``, for estimation of CATE given a binary treatment and set of covariates
  // desc = ``continuous_treatment_effect``, for estimation of CATE given a continuous treatment and set of covariates
  TaskType task = TaskType::kSupervisedLearning;

  // [doc-only]
  // type = enum
  // options = continuous, binary, ordinal
  // desc = outcome type
  // desc = ``continuous``, outcome can take any values in (-inf, inf)
  // desc = ``binary``, outcome can take values {0, 1}
  // desc = ``ordinal``, outcome can take discrete set of integer values for which order is informative
  OutcomeType outcome_type = OutcomeType::kContinuous;

  // [doc-only]
  // type = enum
  // options = xbart, bart, warm_start
  // desc = sampling method type
  // desc = ``xbart``, `Accelerated Bayesian Additive Regression Trees <https://www.tandfonline.com/doi/abs/10.1080/01621459.2021.1942012>`__
  // desc = ``bart``, `Bayesian Additive Regression Trees <https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full>`__
  // desc = ``warm_start``, BART sampler initialized with XBART
  MethodType method_type = MethodType::kXBART;

  // alias = num_iterations, num_estimators
  // check = >=0
  // desc = number of trees used in each ensemble
  int num_trees = 100;

  // alias = num_draws, mc_draws
  // check = >=0
  // desc = number of samples (not counting burn-in) of a tree ensemble to draw
  int num_samples = 5;

  // alias = n_burn
  // check = >=0
  // desc = number of "burn-in" samples to be drawn and then discarded
  // desc = **Note**: for warm start, this only applies to the first stage XBART sampler (second stage BART has no burn-in)
  int num_burnin = 100;

  // [doc-only]
  // alias = random_seed, random_state
  // desc = seed used to draw split rules and leaf node parameters
  int random_seed = -1;

  // desc = maximum depth of tree models
  // desc = ``<= 0`` means no limit
  int max_depth = -1;

  // alias = min_data_per_leaf, min_data, min_child_samples, min_samples_leaf
  // check = >=0
  // desc = minimal number of data in one leaf. Can be used to deal with over-fitting
  int min_data_in_leaf = 10;

  // check = >0
  // alias = num_classes
  // desc = used only in ``multi-class`` classification application
  int num_class = 1;

  // check = >0
  // desc = First parameter in the BART / XBART node split prior (proportional to alpha*(1+depth)^beta)
  double alpha = 0.9;

  // check = >0
  // desc = Second parameter in the BART / XBART node split prior (proportional to alpha*(1+depth)^beta)
  double beta = 2.;

  // desc = used only in bart regression applications
  // desc = prior mean parameter for the outcome mean term in a BART model
  double mu = 0.;

  // check = >0.0
  // desc = used only in bart regression applications
  // desc = variance scaling parameter for the outcome mean term in a BART model
  double kappa = 5.;

  // check = >0.0
  // desc = used only in xbart regression applications
  // desc = shape parameter for inverse-gamma prior on residual variance
  double a_sigma = 16;

  // check = >0.0
  // desc = used only in xbart regression applications
  // desc = scale parameter for inverse-gamma prior on residual variance
  double b_sigma = 4;

  // check = >0.0
  // desc = used only in xbart regression applications
  // desc = shape parameter for inverse-gamma prior on leaf mean variances
  double a_tau = 3;

  // check = >0.0
  // desc = used only in xbart regression applications
  // desc = scale parameter for inverse-gamma prior on leaf mean variances
  // desc = **Note**: this is reset to 0.5*var(y)/num_trees if 
  // data_driven_prior is set to true
  double b_tau = 0.5;

  // check = >0.0
  // desc = used only in bart regression applications
  // desc = shape parameter for inverse-gamma prior on residual variances 
  // is equal to nu / 2
  double nu = 5;

  // check = >0.0
  // desc = used only in bart regression applications
  // desc = scale parameter for inverse-gamma prior on residual variances
  // is equal to nu*lambda / 2
  // desc = **Note**: this is re-calibrated so that p(sigma < sigma_hat) = 0.9
  // if data_driven_prior is set to true
  double lambda = 5;

  // desc = used only in bart regression applications
  // desc = mean hyperparameter for prior on leaf node parameters
  // desc = **Note**: this is re-calibrated so that 
  // min(y) = num_trees * mu_mean - k sqrt(num_trees) * mu_sigma
  // max(y) = num_trees * mu_mean + k sqrt(num_trees) * mu_sigma
  double mu_mean = 0;

  // check = >0.0
  // desc = used only in bart regression applications
  // desc = variance hyperparameter for prior on leaf node parameters
  // desc = **Note**: this is re-calibrated so that 
  // min(y) = num_trees * mu_mean - k sqrt(num_trees) * mu_sigma
  // max(y) = num_trees * mu_mean + k sqrt(num_trees) * mu_sigma
  double mu_sigma = 1;

  // check = >=1
  // desc = number of cutpoints to consider at each split
  data_size_t cutpoint_grid_size = 100;

  // check = >0.0
  // desc = minimum reduction in SSR required to perform a split
  double min_ssr_reduction = 1E-6;
  
  // desc = whether to set prior parameters based on observed data, 
  // as described in the BART and XBART papers
  bool data_driven_prior = true;

  // desc = whether to load a dataset's column names from the header of an input file
  bool header = true;

  // type = comma-separated ints
  // desc = used to specify indices of outcome(s), in the format "0,2,3"
  std::string outcome_columns = "";

  // type = comma-separated ints
  // desc = used to specify indices of treatment(s), in the format "0,2,3"
  std::string treatment_columns = "";

  // type = comma-separated ints
  // desc = used to specify indices of features that are (unordered) categorical, in the format "0,2,3"
  std::string unordered_categoricals = "";

  // type = comma-separated ints
  // desc = used to specify indices of features that are (ordered) categorical, in the format "0,2,3"
  std::string ordered_categoricals = "";

  // type = comma-separated ints
  // desc = used to specify indices of basis vectors, in the format "0,2,3"
  std::string basis_columns = "";

  // type = int
  // check = >10
  // desc = Number of cutpoints to consider for numeric features with many observations
  int32_t num_cutpoints = 500;

  // type = int or string
  // alias = label
  // desc = used to specify the label column
  // desc = use number for index, e.g. ``label=0`` means column\_0 is the label
  // desc = add a prefix ``name:`` for column name, e.g. ``label=name:is_click``
  // desc = if omitted, the first column in the training data is used as the label
  // desc = **Note**: name prefix works only in case of loading data directly from text file
  std::string label_column = "";

  // type = int or string
  // alias = treatment
  // desc = used to specify the treatment column
  // desc = use number for index, e.g. ``treatment=0`` means column\_0 is the weight
  // desc = add a prefix ``name:`` for column name, e.g. ``treatment=name:treatment``
  // desc = **Note**: name prefix works only in case of loading data directly from text file
  std::string treatment_column = "";

  // desc = use precise floating point number parsing for text parser (e.g. CSV, TSV, LibSVM input)
  // desc = **Note**: setting this to ``true`` may lead to much slower text parsing
  bool precise_float_parser = false;

  // alias = train, train_data, train_data_file, data_filename
  // desc = path of training data, LightGBM will train from this data
  // desc = **Note**: can be used only in CLI version
  std::string data = "";

  // alias = prediction, prediction_data_file, data_filename
  // desc = path of prediction data
  std::string prediction_data = "";

  // alias = verbose
  // desc = controls the level of LightGBM's verbosity
  // desc = ``< 0``: Fatal, ``= 0``: Error (Warning), ``= 1``: Info, ``> 1``: Debug
  int verbosity = 1;

  // desc = indicator of whether or not model draws should be serialized to text files
  bool save_model_draws = false;

  // desc = helper parameter for reading data from a text file
  size_t file_load_progress_interval_bytes = size_t(10) * 1024 * 1024 * 1024;

  STOCHTREE_EXPORT void Set(const std::unordered_map<std::string, std::string>& params);
  static const std::unordered_map<std::string, std::string>& alias_table();
  static const std::unordered_map<std::string, std::vector<std::string>>& parameter2aliases();
  static const std::unordered_set<std::string>& parameter_set();
  std::vector<std::vector<double>> auc_mu_weights_matrix;
  std::vector<std::vector<int>> interaction_constraints_vector;
  static const std::unordered_map<std::string, std::string>& ParameterTypes();
  static const std::string DumpAliases();
  const int GetOutcomeIndex(std::vector<std::string>& variable_names);
  const int GetTreatmentIndex(std::vector<std::string>& variable_names);

 private:
  void CheckParamConflict();
  void GetMembersFromString(const std::unordered_map<std::string, std::string>& params);
  std::string SaveMembersToString() const;
  std::string TaskToStr() const;
  std::string OutcomeTypeToStr() const;
  std::string MethodTypeToStr() const;
};

inline bool Config::GetString(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, std::string* out) {
  if (params.count(name) > 0 && !params.at(name).empty()) {
    *out = params.at(name);
    return true;
  }
  return false;
}

inline bool Config::GetInt(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, int* out) {
  if (params.count(name) > 0 && !params.at(name).empty()) {
    if (!Common::AtoiAndCheck(params.at(name).c_str(), out)) {
      Log::Fatal("Parameter %s should be of type int, got \"%s\"",
                 name.c_str(), params.at(name).c_str());
    }
    return true;
  }
  return false;
}

inline bool Config::GetDouble(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, double* out) {
  if (params.count(name) > 0 && !params.at(name).empty()) {
    if (!Common::AtofAndCheck(params.at(name).c_str(), out)) {
      Log::Fatal("Parameter %s should be of type double, got \"%s\"",
                 name.c_str(), params.at(name).c_str());
    }
    return true;
  }
  return false;
}

inline bool Config::GetBool(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, bool* out) {
  if (params.count(name) > 0 && !params.at(name).empty()) {
    std::string value = params.at(name);
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("false") || value == std::string("-")) {
      *out = false;
    } else if (value == std::string("true") || value == std::string("+")) {
      *out = true;
    } else {
      Log::Fatal("Parameter %s should be \"true\"/\"+\" or \"false\"/\"-\", got \"%s\"",
                 name.c_str(), params.at(name).c_str());
    }
    return true;
  }
  return false;
}

inline bool Config::SortAlias(const std::string& x, const std::string& y) {
  return x.size() < y.size() || (x.size() == y.size() && x < y);
}

struct ParameterAlias {
  static void KeyAliasTransform(std::unordered_map<std::string, std::string>* params) {
    std::unordered_map<std::string, std::string> tmp_map;
    for (const auto& pair : *params) {
      auto alias = Config::alias_table().find(pair.first);
      if (alias != Config::alias_table().end()) {  // found alias
        auto alias_set = tmp_map.find(alias->second);
        if (alias_set != tmp_map.end()) {  // alias already set
          if (Config::SortAlias(alias_set->second, pair.first)) {
            Log::Warning("%s is set with %s=%s, %s=%s will be ignored. Current value: %s=%s",
                         alias->second.c_str(), alias_set->second.c_str(), params->at(alias_set->second).c_str(),
                         pair.first.c_str(), pair.second.c_str(), alias->second.c_str(), params->at(alias_set->second).c_str());
          } else {
            Log::Warning("%s is set with %s=%s, will be overridden by %s=%s. Current value: %s=%s",
                         alias->second.c_str(), alias_set->second.c_str(), params->at(alias_set->second).c_str(),
                         pair.first.c_str(), pair.second.c_str(), alias->second.c_str(), pair.second.c_str());
            tmp_map[alias->second] = pair.first;
          }
        } else {  // alias not set
          tmp_map.emplace(alias->second, pair.first);
        }
      } else if (Config::parameter_set().find(pair.first) == Config::parameter_set().end()) {
        Log::Warning("Unknown parameter: %s", pair.first.c_str());
      }
    }
    for (const auto& pair : tmp_map) {
      auto alias = params->find(pair.first);
      if (alias == params->end()) {  // not find
        params->emplace(pair.first, params->at(pair.second));
        params->erase(pair.second);
      } else {
        Log::Warning("%s is set=%s, %s=%s will be ignored. Current value: %s=%s",
                     pair.first.c_str(), alias->second.c_str(), pair.second.c_str(), params->at(pair.second).c_str(),
                     pair.first.c_str(), alias->second.c_str());
      }
    }
  }
};

}   // namespace StochTree

#endif   // STOCHTREE_CONFIG_H_
