/*! Copyright (c) 2024 by stochtree authors */
#include <Eigen/Dense>
#include <stochtree/container.h>
#include <stochtree/data.h>

namespace StochTree {

ForestContainer::ForestContainer(int num_trees, int output_dimension, bool is_leaf_constant) {
  forests_ = std::vector<std::unique_ptr<TreeEnsemble>>(0);
  num_samples_ = 0;
  num_trees_ = num_trees;
  output_dimension_ = output_dimension;
  is_leaf_constant_ = is_leaf_constant;
  initialized_ = true;
}

ForestContainer::ForestContainer(int num_samples, int num_trees, int output_dimension, bool is_leaf_constant) {
  forests_ = std::vector<std::unique_ptr<TreeEnsemble>>(num_samples);
  for (auto& forest : forests_) {
    forest.reset(new TreeEnsemble(num_trees, output_dimension, is_leaf_constant));
  }
  num_samples_ = num_samples;
  num_trees_ = num_trees;
  output_dimension_ = output_dimension;
  is_leaf_constant_ = is_leaf_constant;
}

void ForestContainer::CopyFromPreviousSample(int new_sample_id, int previous_sample_id) {
  forests_[new_sample_id].reset(new TreeEnsemble(*forests_[previous_sample_id]));
}

void ForestContainer::AddSamples(int num_samples) {
  CHECK(initialized_);
  int total_new_samples = num_samples + num_samples_;
  forests_.resize(total_new_samples);
  for (int i = num_samples_; i < total_new_samples; i++) {
    forests_[i].reset(new TreeEnsemble(num_trees_, output_dimension_, is_leaf_constant_));
  }
  num_samples_ = total_new_samples;
}

std::vector<double> ForestContainer::Predict(ForestDataset& dataset) {
  data_size_t n = dataset.NumObservations();
  data_size_t total_output_size = n*num_samples_;
  std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  for (int i = 0; i < num_samples_; i++) {
    auto num_trees = forests_[i]->NumTrees();
    forests_[i]->PredictInplace(dataset, output, 0, num_trees, offset);
    offset += n;
  }
  return output;
}

std::vector<double> ForestContainer::PredictRaw(ForestDataset& dataset) {
  data_size_t n = dataset.NumObservations();
  data_size_t total_output_size = n * output_dimension_ * num_samples_;
  std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  for (int i = 0; i < num_samples_; i++) {
    auto num_trees = forests_[i]->NumTrees();
    forests_[i]->PredictRawInplace(dataset, output, 0, num_trees, offset);
    offset += n * output_dimension_;
  }
  return output;
}

std::vector<double> ForestContainer::PredictRaw(ForestDataset& dataset, int forest_num) {
  data_size_t n = dataset.NumObservations();
  data_size_t total_output_size = n * output_dimension_;
  std::vector<double> output(total_output_size);
  data_size_t offset = 0;
  auto num_trees = forests_[forest_num]->NumTrees();
  forests_[forest_num]->PredictRawInplace(dataset, output, 0, num_trees, offset);
  return output;
}

/*! \brief Save to JSON */
json11::Json ForestContainer::to_json() {
  json11::Json::object result_obj;
  result_obj.emplace(std::make_pair("num_samples", this->num_samples_));
  result_obj.emplace(std::make_pair("num_trees", this->num_trees_));
  result_obj.emplace(std::make_pair("output_dimension", this->output_dimension_));
  result_obj.emplace(std::make_pair("is_leaf_constant", this->is_leaf_constant_));
  result_obj.emplace(std::make_pair("initialized", this->initialized_));

  std::string forest_label;
  for (int i = 0; i < forests_.size(); i++) {
    forest_label = "forest_" + std::to_string(i);
    result_obj.emplace(std::make_pair(forest_label, forests_[i]->to_json()));
  }
  
  json11::Json result(result_obj);
  return result;
}

/*! \brief Load from JSON */
void ForestContainer::from_json(const json11::Json& json_forest_container) {
  this->num_samples_ = json_forest_container["num_samples"].int_value();
  this->num_trees_ = json_forest_container["num_trees"].int_value();
  this->output_dimension_ = json_forest_container["output_dimension"].int_value();
  this->is_leaf_constant_ = json_forest_container["is_leaf_constant"].bool_value();
  this->initialized_ = json_forest_container["initialized"].bool_value();

  std::string forest_label;
  forests_.clear();
  forests_.resize(this->num_samples_);
  for (int i = 0; i < this->num_samples_; i++) {
    forest_label = "forest_" + std::to_string(i);
    forests_[i] = std::make_unique<TreeEnsemble>();
    forests_[i]->from_json(json_forest_container[forest_label]);
  }
}

} // namespace StochTree
