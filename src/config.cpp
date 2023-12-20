/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <stochtree/config.h>
#include <stochtree/common.h>
#include <stochtree/log.h>
#include <stochtree/random.h>

#include <limits>

namespace StochTree {

std::vector<int> Config::Str2FeatureVec(const char* parameters) {
  std::vector<int> feature_vec;
  auto args = Common::Split(parameters, ",");
  for (auto arg : args) {
    FeatureUnpack(&feature_vec, Common::Trim(arg).c_str());
  }
  return feature_vec;
}

void Config::FeatureUnpack(std::vector<int>* categorical_variables, const char* var_id) {
  std::string var_clean = Common::RemoveQuotationSymbol(Common::Trim(var_id));
  int out;
  bool success = Common::AtoiAndCheck(var_clean.c_str(), &out);
  if (success) {
    categorical_variables->push_back(out);
  } else {
    Log::Warning("Parsed variable index %s cannot be cast to an integer", var_clean.c_str());
  }
}

void Config::KV2Map(std::unordered_map<std::string, std::vector<std::string>>* params, const char* kv) {
  std::vector<std::string> tmp_strs = Common::Split(kv, '=');
  if (tmp_strs.size() == 2 || tmp_strs.size() == 1) {
    std::string key = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[0]));
    std::string value = "";
    if (tmp_strs.size() == 2) {
      value = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[1]));
    }
    if (key.size() > 0) {
      params->operator[](key).emplace_back(value);
    }
  } else {
    Log::Warning("Unknown parameter %s", kv);
  }
}

void GetFirstValueAsInt(const std::unordered_map<std::string, std::vector<std::string>>& params, std::string key, int* out) {
  const auto pair = params.find(key);
  if (pair != params.end()) {
    auto candidate = pair->second[0].c_str();
    if (!Common::AtoiAndCheck(candidate, out)) {
      Log::Fatal("Parameter %s should be of type int, got \"%s\"", key.c_str(), candidate);
    }
  }
}

void Config::SetVerbosity(const std::unordered_map<std::string, std::vector<std::string>>& params) {
  int verbosity = Config().verbosity;
  GetFirstValueAsInt(params, "verbose", &verbosity);
  GetFirstValueAsInt(params, "verbosity", &verbosity);
  if (verbosity < 0) {
    StochTree::Log::ResetLogLevel(StochTree::LogLevel::Fatal);
  } else if (verbosity == 0) {
    StochTree::Log::ResetLogLevel(StochTree::LogLevel::Warning);
  } else if (verbosity == 1) {
    StochTree::Log::ResetLogLevel(StochTree::LogLevel::Info);
  } else {
    StochTree::Log::ResetLogLevel(StochTree::LogLevel::Debug);
  }
}

void Config::KeepFirstValues(const std::unordered_map<std::string, std::vector<std::string>>& params, std::unordered_map<std::string, std::string>* out) {
  for (auto pair = params.begin(); pair != params.end(); ++pair) {
    auto name = pair->first.c_str();
    auto values = pair->second;
    out->emplace(name, values[0]);
    for (size_t i = 1; i < pair->second.size(); ++i) {
      Log::Warning("%s is set=%s, %s=%s will be ignored. Current value: %s=%s",
        name, values[0].c_str(),
        name, values[i].c_str(),
        name, values[0].c_str());
    }
  }
}

std::unordered_map<std::string, std::string> Config::Str2Map(const char* parameters) {
  std::unordered_map<std::string, std::vector<std::string>> all_params;
  std::unordered_map<std::string, std::string> params;
  auto args = Common::Split(parameters, " \t\n\r");
  for (auto arg : args) {
    KV2Map(&all_params, Common::Trim(arg).c_str());
  }
  SetVerbosity(all_params);
  KeepFirstValues(all_params, &params);
  ParameterAlias::KeyAliasTransform(&params);
  return params;
}

void GetBoostingType(const std::unordered_map<std::string, std::string>& params, std::string* boosting) {
  std::string value;
  if (Config::GetString(params, "boosting", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("gbdt") || value == std::string("gbrt")) {
      *boosting = "gbdt";
    } else if (value == std::string("dart")) {
      *boosting = "dart";
    } else if (value == std::string("goss")) {
      *boosting = "goss";
    } else if (value == std::string("rf") || value == std::string("random_forest")) {
      *boosting = "rf";
    } else {
      Log::Fatal("Unknown boosting type %s", value.c_str());
    }
  }
}

void GetDataSampleStrategy(const std::unordered_map<std::string, std::string>& params, std::string* strategy) {
  std::string value;
  if (Config::GetString(params, "data_sample_strategy", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("goss")) {
      *strategy = "goss";
    } else if (value == std::string("bagging")) {
      *strategy = "bagging";
    } else {
      Log::Fatal("Unknown sample strategy %s", value.c_str());
    }
  }
}

void GetTaskType(const std::unordered_map<std::string, std::string>& params, TaskType* task) {
  std::string value;
  if (Config::GetString(params, "task", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("supervised_learning")) {
      *task = TaskType::kSupervisedLearning;
    } else if (value == std::string("binary_treatment_effect")) {
      *task = TaskType::kBinaryTreatmentEffect;
    } else if (value == std::string("continuous_treatment_effect")) {
      *task = TaskType::kContinuousTreatmentEffect;
    } else {
      Log::Fatal("Unknown task type %s", value.c_str());
    }
  }
}

void GetOutcomeType(const std::unordered_map<std::string, std::string>& params, OutcomeType* outcome_type) {
  std::string value;
  if (Config::GetString(params, "outcome_type", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("continuous")) {
      *outcome_type = OutcomeType::kContinuous;
    } else if (value == std::string("binary")) {
      *outcome_type = OutcomeType::kBinary;
    } else if (value == std::string("ordinal")) {
      *outcome_type = OutcomeType::KOrdinal;
    } else {
      Log::Fatal("Unknown outcome type %s", value.c_str());
    }
  }
}

void GetMethodType(const std::unordered_map<std::string, std::string>& params, MethodType* method_type) {
  std::string value;
  if (Config::GetString(params, "method_type", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("xbart")) {
      *method_type = MethodType::kXBART;
    } else if (value == std::string("bart")) {
      *method_type = MethodType::kBART;
    } else if (value == std::string("warm_start")) {
      *method_type = MethodType::kWarmStartBART;
    } else {
      Log::Fatal("Unknown method type %s", value.c_str());
    }
  }
}

void Config::Set(const std::unordered_map<std::string, std::string>& params) {
  // Set seed and initialize a random number generator
  if (GetInt(params, "random_seed", &random_seed)) {
    Random rand(random_seed);
  }

  GetTaskType(params, &task);
  GetOutcomeType(params, &outcome_type);
  GetMethodType(params, &method_type);
  GetMembersFromString(params);
  CheckParamConflict();
}

const int Config::GetOutcomeIndex(std::vector<std::string>& variable_names) {
  int label_idx_ = NO_SPECIFIC;
  std::string name_prefix("name:");
  // determine label index
  if (label_column.size() > 0) {
    if (Common::StartsWith(label_column, name_prefix)) {
      std::string name = label_column.substr(name_prefix.size());
      label_idx_ = -1;
      for (int i = 0; i < static_cast<int>(variable_names.size()); ++i) {
        if (name == variable_names[i]) {
          label_idx_ = i;
          break;
        }
      }
      if (label_idx_ >= 0) {
        Log::Info("Using column %s as label", name.c_str());
      } else {
        Log::Fatal("Could not find label column %s in data file \n"
                    "or data file doesn't contain header", name.c_str());
      }
    } else {
      if (!Common::AtoiAndCheck(label_column.c_str(), &label_idx_)) {
        Log::Fatal("label_column is not a number,\n"
                    "if you want to use a column name,\n"
                    "please add the prefix \"name:\" to the column name");
      }
      Log::Info("Using column number %d as label", label_idx_);
    }
  }
  return label_idx_;
}

const int Config::GetTreatmentIndex(std::vector<std::string>& variable_names) {
  int treatment_idx_ = NO_SPECIFIC;
  std::string name_prefix("name:");
  // determine label index
  if (treatment_column.size() > 0) {
    if (Common::StartsWith(treatment_column, name_prefix)) {
      std::string name = treatment_column.substr(name_prefix.size());
      treatment_idx_ = -1;
      for (int i = 0; i < static_cast<int>(variable_names.size()); ++i) {
        if (name == variable_names[i]) {
          treatment_idx_ = i;
          break;
        }
      }
      if (treatment_idx_ >= 0) {
        Log::Info("Using column %s as treatment", name.c_str());
      } else {
        Log::Fatal("Could not find treatment column %s in data file \n"
                    "or data file doesn't contain header", name.c_str());
      }
    } else {
      if (!Common::AtoiAndCheck(treatment_column.c_str(), &treatment_idx_)) {
        Log::Fatal("treatment_column is not a number,\n"
                    "if you want to use a column name,\n"
                    "please add the prefix \"name:\" to the column name");
      }
      Log::Info("Using column number %d as treatment", treatment_idx_);
    }
  }
  return treatment_idx_;
}

void Config::CheckParamConflict() {}

std::string Config::TaskToStr() const {
  if (task == TaskType::kSupervisedLearning){
    return std::string("supervised_learning");
  } else if (task == TaskType::kBinaryTreatmentEffect){
    return std::string("binary_treatment_effect");
  } else if (task == TaskType::kContinuousTreatmentEffect){
    return std::string("continuous_treatment_effect");
  } else {
    Log::Fatal("Unknown task type %s", task);
  }
}

std::string Config::OutcomeTypeToStr() const {
  if (outcome_type == OutcomeType::kContinuous){
    return std::string("continuous");
  } else if (outcome_type == OutcomeType::kBinary){
    return std::string("binary");
  } else if (outcome_type == OutcomeType::KOrdinal){
    return std::string("ordinal");
  } else {
    Log::Fatal("Unknown outcome type %s", outcome_type);
  }
}

std::string Config::MethodTypeToStr() const {
  if (method_type == MethodType::kXBART){
    return std::string("xbart");
  } else if (method_type == MethodType::kBART){
    return std::string("bart");
  } else if (method_type == MethodType::kWarmStartBART){
    return std::string("warm_start");
  } else {
    Log::Fatal("Unknown method type %s", method_type);
  }
}

std::string Config::ToString() const {
  std::stringstream str_buf;
  str_buf << "[task: " << TaskToStr() << "]\n";
  str_buf << "[outcome_type: " << OutcomeTypeToStr() << "]\n";
  str_buf << "[method_type: " << MethodTypeToStr() << "]\n";
  str_buf << SaveMembersToString();
  return str_buf.str();
}

const std::string Config::DumpAliases() {
  auto map = Config::parameter2aliases();
  for (auto& pair : map) {
    std::sort(pair.second.begin(), pair.second.end(), SortAlias);
  }
  std::stringstream str_buf;
  str_buf << "{\n";
  bool first = true;
  for (const auto& pair : map) {
    if (first) {
      str_buf << "   \"";
      first = false;
    } else {
      str_buf << "   , \"";
    }
    str_buf << pair.first << "\": [";
    if (pair.second.size() > 0) {
      str_buf << "\"" << CommonC::Join(pair.second, "\", \"") << "\"";
    }
    str_buf << "]\n";
  }
  str_buf << "}\n";
  return str_buf.str();
}

std::string Config::SaveMembersToString() const {
  std::stringstream str_buf;
  str_buf << "[data: " << data << "]\n";
  str_buf << "[prediction_data: " << prediction_data << "]\n";
  str_buf << "[num_trees: " << num_trees << "]\n";
  str_buf << "[num_samples: " << num_samples << "]\n";
  str_buf << "[num_burnin: " << num_burnin << "]\n";
  str_buf << "[random_seed: " << random_seed << "]\n";
  str_buf << "[max_depth: " << max_depth << "]\n";
  str_buf << "[min_data_in_leaf: " << min_data_in_leaf << "]\n";
  str_buf << "[num_class: " << num_class << "]\n";
  str_buf << "[alpha: " << alpha << "]\n";
  str_buf << "[beta: " << beta << "]\n";
  str_buf << "[mu: " << mu << "]\n";
  str_buf << "[kappa: " << kappa << "]\n";
  str_buf << "[a_sigma: " << a_sigma << "]\n";
  str_buf << "[b_sigma: " << b_sigma << "]\n";
  str_buf << "[a_tau: " << a_tau << "]\n";
  str_buf << "[b_tau: " << b_tau << "]\n";
  str_buf << "[nu: " << nu << "]\n";
  str_buf << "[lambda: " << lambda << "]\n";
  str_buf << "[mu_mean: " << mu_mean << "]\n";
  str_buf << "[mu_sigma: " << mu_sigma << "]\n";
  str_buf << "[data_driven_prior: " << data_driven_prior << "]\n";
  str_buf << "[min_ssr_reduction: " << min_ssr_reduction << "]\n";
  str_buf << "[header: " << header << "]\n";
  str_buf << "[unordered_categoricals: " << unordered_categoricals << "]\n";
  str_buf << "[ordered_categoricals: " << ordered_categoricals << "]\n";
  str_buf << "[basis_columns: " << basis_columns << "]\n";
  str_buf << "[cutpoint_grid_size: " << cutpoint_grid_size << "]\n";
  str_buf << "[outcome_columns: " << outcome_columns << "]\n";
  str_buf << "[treatment_columns: " << treatment_columns << "]\n";
  str_buf << "[save_model_draws: " << save_model_draws << "]\n";
  str_buf << "[precise_float_parser: " << precise_float_parser << "]\n";
  return str_buf.str();
}

const std::unordered_map<std::string, std::string>& Config::alias_table() {
  static std::unordered_map<std::string, std::string> aliases({
    {"task_type", "task"}, 
    {"target_type", "outcome_type"}, 
    {"sampling_method", "method_type"}, 
    {"data_filename", "data"}, 
    {"prediction_data_filename", "prediction_data"}, 
    {"num_estimators", "num_trees"}, 
    {"num_draws", "num_samples"}, 
    {"num_discarded_samples", "num_burnin"}, 
    {"seed", "random_seed"}, 
    {"random_state", "random_seed"}, 
  });
  return aliases;
}


const std::unordered_set<std::string>& Config::parameter_set() {
  static std::unordered_set<std::string> params({
    "task", "outcome_type", "method_type", "data", "prediction_data", 
    "num_class", "num_trees", "num_samples", "num_burnin", 
    "random_seed", "max_depth", "min_data_in_leaf", 
    "mu", "kappa", "a_sigma", "b_sigma", "a_tau", "b_tau", "alpha", "beta", 
    "nu", "lambda", "mu_mean", "mu_sigma", "cutpoint_grid_size", "data_driven_prior", 
    "min_ssr_reduction", "header", "unordered_categoricals", "ordered_categoricals", 
    "basis_columns", "outcome_columns", "treatment_columns", 
    "label_column", "treatment_column", "save_model_draws", "precise_float_parser",
  });
  return params;
}

void Config::GetMembersFromString(const std::unordered_map<std::string, std::string>& params) {
  std::string tmp_str = "";
  GetString(params, "data", &data);
  GetString(params, "prediction_data", &prediction_data);

  GetInt(params, "num_class", &num_class);

  GetInt(params, "num_trees", &num_trees);
  CHECK_GE(num_trees, 0);

  GetInt(params, "num_samples", &num_samples);
  CHECK_GT(num_samples, 0);

  GetInt(params, "num_burnin", &num_burnin);
  CHECK_GE(num_burnin, 0);

  GetInt(params, "max_depth", &max_depth);
  // CHECK_GT(max_depth, 0);

  GetInt(params, "min_data_in_leaf", &min_data_in_leaf);
  CHECK_GE(min_data_in_leaf, 0);

  GetInt(params, "random_seed", &random_seed);

  GetDouble(params, "mu", &mu);

  GetDouble(params, "kappa", &kappa);
  CHECK_GT(kappa, 0.0);

  GetDouble(params, "a_sigma", &a_sigma);
  CHECK_GT(a_sigma, 0.0);

  GetDouble(params, "b_sigma", &b_sigma);
  CHECK_GT(b_sigma, 0.0);

  GetDouble(params, "a_tau", &a_tau);
  CHECK_GT(a_tau, 0.0);

  GetDouble(params, "b_tau", &b_tau);
  CHECK_GT(b_tau, 0.0);

  GetDouble(params, "alpha", &alpha);
  CHECK_GT(alpha, 0.0);
  CHECK_LT(alpha, 1.0);

  GetDouble(params, "beta", &beta);
  CHECK_GT(beta, 0.0);

  GetDouble(params, "nu", &nu);
  CHECK_GT(nu, 0.0);

  GetDouble(params, "lambda", &lambda);
  CHECK_GT(lambda, 0.0);

  GetDouble(params, "mu_mean", &mu_mean);
  
  GetDouble(params, "mu_sigma", &mu_sigma);
  CHECK_GT(mu_sigma, 0.0);

  GetInt(params, "cutpoint_grid_size", &cutpoint_grid_size);
  CHECK_GT(cutpoint_grid_size, 0.0);
  
  GetBool(params, "data_driven_prior", &data_driven_prior);
  
  GetDouble(params, "min_ssr_reduction", &min_ssr_reduction);
  CHECK_GT(min_ssr_reduction, 0.0);

  GetBool(params, "header", &header);

  GetString(params, "unordered_categoricals", &unordered_categoricals);

  GetString(params, "ordered_categoricals", &ordered_categoricals);

  GetString(params, "basis_columns", &basis_columns);

  GetString(params, "outcome_columns", &outcome_columns);

  GetString(params, "treatment_columns", &treatment_columns);

  GetString(params, "label_column", &label_column);

  GetString(params, "treatment_column", &treatment_column);

  GetBool(params, "save_model_draws", &save_model_draws);

  GetBool(params, "precise_float_parser", &precise_float_parser);
  CHECK_GT(num_class, 0);

}

const std::unordered_map<std::string, std::vector<std::string>>& Config::parameter2aliases() {
  static std::unordered_map<std::string, std::vector<std::string>> map({
    {"task", {"task_type"}},
    {"outcome_type", {"target_type"}},
    {"method_type", {"sampling_method"}},
    {"data", {"data_filename"}},
    {"prediction_data", {"prediction_data_filename"}},
    {"num_class", {}},
    {"num_trees", {"num_estimators"}},
    {"num_samples", {"num_draws"}},
    {"num_burnin", {"num_discarded_samples"}},
    {"random_seed", {"seed", "random_state"}},
    {"max_depth", {}},
    {"min_data_in_leaf", {}},
    {"mu", {}},
    {"kappa", {}},
    {"alpha", {}},
    {"beta", {}},
    {"a_sigma", {}},
    {"b_sigma", {}},
    {"a_tau", {}},
    {"b_tau", {}},
    {"nu", {}},
    {"lambda", {}},
    {"mu_mean", {}},
    {"mu_sigma", {}},
    {"cutpoint_grid_size", {}},
    {"data_driven_prior", {}},
    {"min_ssr_reduction", {}},
    {"header", {}},
    {"unordered_categoricals", {}},
    {"ordered_categoricals", {}},
    {"basis_columns", {}},
    {"outcome_columns", {}},
    {"treatment_columns", {}},
    {"label_column", {}},
    {"treatment_column", {}},
    {"save_model_draws", {}},
    {"precise_float_parser", {}},
  });
  return map;
}

const std::unordered_map<std::string, std::string>& Config::ParameterTypes() {
  static std::unordered_map<std::string, std::string> map({
    {"task", "string"},
    {"outcome_type", "string"},
    {"method_type", "string"},
    {"data", "string"},
    {"prediction_data", "string"},
    {"data_source", "string"},
    {"num_class", "int"},
    {"num_trees", "int"},
    {"num_samples", "int"},
    {"num_burnin", "int"},
    {"random_seed", "int"},
    {"max_depth", "int"},
    {"min_data_in_leaf", "int"},
    {"mu", "double"},
    {"kappa", "double"},
    {"alpha", "double"},
    {"beta", "double"},
    {"a_sigma", "double"},
    {"b_sigma", "double"},
    {"a_tau", "double"},
    {"b_tau", "double"},
    {"nu", "double"},
    {"lambda", "double"},
    {"mu_mean", "double"},
    {"mu_sigma", "double"},
    {"cutpoint_grid_size", "int"},
    {"data_driven_prior", "bool"},
    {"min_ssr_reduction", "double"},
    {"header", "bool"},
    {"unordered_categoricals", "string"},
    {"ordered_categoricals", "string"},
    {"basis_columns", "string"},
    {"outcome_columns", "string"},
    {"treatment_columns", "string"},
    {"label_column", "string"},
    {"treatment_column", "string"},
    {"save_model_draws", "bool"},
    {"precise_float_parser", "bool"},
  });
  return map;
}

}  // namespace StochTree
