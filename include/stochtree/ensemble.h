/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * Inspired by the design of the Learner, GBTreeModel, and GBTree classes in xgboost, 
 * released under the Apache license with the following copyright:
 * 
 * Copyright 2015-2023 by XGBoost Contributors
 */
#ifndef STOCHTREE_ENSEMBLE_H_
#define STOCHTREE_ENSEMBLE_H_

#include <stochtree/data.h>
#include <stochtree/tree.h>

#include <algorithm>
#include <deque>
#include <random>
#include <unordered_map>

namespace StochTree {

class TreeEnsemble {
 public:
  TreeEnsemble(int num_trees, int output_dimension = 1, bool is_leaf_constant = true) {
    // Initialize trees in the ensemble
    trees_ = std::vector<std::unique_ptr<Tree>>(num_trees);
    for (int i = 0; i < num_trees; i++) {
      trees_[i].reset(new Tree());
      trees_[i]->Init(output_dimension);
    }
    // Store ensemble configurations
    num_trees_ = num_trees;
    output_dimension_ = output_dimension;
    is_leaf_constant_ = is_leaf_constant;
  }
  TreeEnsemble(TreeEnsemble& ensemble) {
    // Unpack ensemble configurations
    num_trees_ = ensemble.num_trees_;
    output_dimension_ = ensemble.output_dimension_;
    is_leaf_constant_ = ensemble.is_leaf_constant_;
    // Initialize trees in the ensemble
    trees_ = std::vector<std::unique_ptr<Tree>>(num_trees_);
    for (int i = 0; i < num_trees_; i++) {
      trees_[i].reset(new Tree());
      // trees_[i]->Init(output_dimension);
    }
    // Clone trees in the ensemble
    // trees_ = std::vector<std::unique_ptr<Tree>>(num_trees_);
    for (int j = 0; j < num_trees_; j++) {
      Tree* tree = ensemble.GetTree(j);
      this->CloneFromExistingTree(j, tree);
    }
  }
  ~TreeEnsemble() {}

  inline Tree* GetTree(int i) {
    return trees_[i].get();
  }

  inline void ResetTree(int i) {
    trees_[i].reset(new Tree());
  }

  inline void ResetInitTree(int i) {
    trees_[i].reset(new Tree());
    trees_[i]->Init(output_dimension_);
  }

  inline void CopyTree(int i, Tree* tree) {
    return trees_[i].reset(tree->Clone());
  }

  inline void CloneFromExistingTree(int i, Tree* tree) {
    return trees_[i]->CloneFromTree(tree);
  }

  inline void PredictInplace(ForestDataset& dataset, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(dataset, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(ForestDataset& dataset, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {
    if (dataset.HasBasis()) {
      CHECK(!is_leaf_constant_);
      PredictInplace(dataset.GetCovariates(), dataset.GetBasis(), output, tree_begin, tree_end, offset);
    } else {
      CHECK(is_leaf_constant_);
      PredictInplace(dataset.GetCovariates(), output, tree_begin, tree_end, offset);
    }
  }

  inline void PredictInplace(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(covariates, basis, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(Eigen::MatrixXd& covariates, Eigen::MatrixXd& basis, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {
    double pred;
    CHECK_EQ(covariates.rows(), basis.rows());
    CHECK_EQ(output_dimension_, trees_[0]->OutputDimension());
    CHECK_EQ(output_dimension_, basis.cols());
    data_size_t n = covariates.rows();
    data_size_t total_output_size = n;
    if (output.size() < total_output_size + offset) {
      Log::Fatal("Mismatched size of prediction vector and training data");
    }
    for (data_size_t i = 0; i < n; i++) {
      pred = 0.0;
      for (size_t j = tree_begin; j < tree_end; j++) {
        auto &tree = *trees_[j];
        std::int32_t nidx = EvaluateTree(tree, covariates, i);
        for (int32_t k = 0; k < output_dimension_; k++) {
          pred += tree.LeafValue(nidx, k) * basis(i, k);
        }
      }
      output[i + offset] = pred;
    }
  }

  inline void PredictInplace(Eigen::MatrixXd& covariates, std::vector<double> &output, data_size_t offset = 0) {
    PredictInplace(covariates, output, 0, trees_.size(), offset);
  }

  inline void PredictInplace(Eigen::MatrixXd& covariates, std::vector<double> &output, int tree_begin, int tree_end, data_size_t offset = 0) {
    double pred;
    data_size_t n = covariates.rows();
    data_size_t total_output_size = n;
    if (output.size() < total_output_size + offset) {
      Log::Fatal("Mismatched size of prediction vector and training data");
    }
    for (data_size_t i = 0; i < n; i++) {
      pred = 0.0;
      for (size_t j = tree_begin; j < tree_end; j++) {
        auto &tree = *trees_[j];
        std::int32_t nidx = EvaluateTree(tree, covariates, i);
        pred += tree.LeafValue(nidx, 0);
      }
      output[i + offset] = pred;
    }
  }

  inline void PredictRawInplace(ForestDataset& dataset, std::vector<double> &output, data_size_t offset = 0) {
    PredictRawInplace(dataset, output, 0, trees_.size(), offset);
  }

  inline void PredictRawInplace(ForestDataset& dataset, std::vector<double> &output, 
                             int tree_begin, int tree_end, data_size_t offset = 0) {
    double pred;
    Eigen::MatrixXd covariates = dataset.GetCovariates();
    CHECK_EQ(output_dimension_, trees_[0]->OutputDimension());
    data_size_t n = covariates.rows();
    data_size_t total_output_size = n * output_dimension_;
    if (output.size() < total_output_size + offset) {
      Log::Fatal("Mismatched size of raw prediction vector and training data");
    }
    for (data_size_t i = 0; i < n; i++) {
      for (int32_t k = 0; k < output_dimension_; k++) {
        pred = 0.0;
        for (size_t j = tree_begin; j < tree_end; j++) {
          auto &tree = *trees_[j];
          int32_t nidx = EvaluateTree(tree, covariates, i);
          pred += tree.LeafValue(nidx, k);
        }
        output[i*output_dimension_ + k + offset] = pred;
      }
    }
  }

  inline int32_t NumTrees() {
    return num_trees_;
  }

  inline int32_t NumLeaves() {
    int32_t result = 0;
    for (int i = 0; i < num_trees_; i++) {
      result += trees_[i]->NumLeaves();
    }
    return result;
  }

  inline double SumLeafSquared() {
    double result = 0.;
    for (int i = 0; i < num_trees_; i++) {
      result += trees_[i]->NumLeaves();
    }
    return result;
  }

  inline int32_t OutputDimension() {
    return output_dimension_;
  }

  inline bool IsLeafConstant() {
    return is_leaf_constant_;
  }

  /*! \brief Save to JSON */
  json11::Json to_json() {
    json11::Json::object result_obj;
    result_obj.emplace(std::make_pair("num_trees", this->num_trees_));
    result_obj.emplace(std::make_pair("output_dimension", this->output_dimension_));
    result_obj.emplace(std::make_pair("is_leaf_constant", this->is_leaf_constant_));

    std::string tree_label;
    for (int i = 0; i < trees_.size(); i++) {
      tree_label = "tree_" + std::to_string(i);
      result_obj.emplace(std::make_pair(tree_label, trees_[i]->to_json()));
    }
    
    json11::Json result(result_obj);
    return result;
  }
  
  /*! \brief Load from JSON */
  void from_json(const json11::Json& json_ensemble) {
    this->num_trees_ = json_ensemble["num_trees"].int_value();
    this->output_dimension_ = json_ensemble["output_dimension"].int_value();
    this->is_leaf_constant_ = json_ensemble["is_leaf_constant"].bool_value();

    std::string tree_label;
    trees_.clear();
    trees_.resize(this->num_trees_);
    for (int i = 0; i < this->num_trees_; i++) {
      tree_label = "tree_" + std::to_string(i);
      trees_[i] = std::make_unique<Tree>();
      trees_[i]->from_json(json_ensemble[tree_label]);
    }
  }

 private:
  std::vector<std::unique_ptr<Tree>> trees_;
  int num_trees_;
  int output_dimension_;
  bool is_leaf_constant_;
};

} // namespace StochTree

#endif // STOCHTREE_ENSEMBLE_H_
