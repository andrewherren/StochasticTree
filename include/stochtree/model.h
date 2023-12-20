/*!
 * Copyright (c) 2023 stochtree authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * Inspired by the design of the Learner, GBTreeModel, and GBTree classes in xgboost, 
 * released under the Apache license with the following copyright:
 * 
 * Copyright 2015-2023 by XGBoost Contributors
 */
#ifndef STOCHTREE_MODEL_H_
#define STOCHTREE_MODEL_H_

#include <stochtree/config.h>
#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/log.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree.h>
#include "../../dependencies/boost_math/include/boost/math/special_functions/gamma.hpp"
#include <Eigen/Dense>

#include <cmath>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace StochTree {

class Model {
 public:
  Model() = default;
  Model(const Config& config) {}
  virtual ~Model() = default;
  virtual void InitializeGlobalParameters(Dataset* dataset) {}
  virtual void SampleTree(Dataset* dataset, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num) {}
  virtual void SampleTree(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num) {}
  virtual void SampleLeafParameters(Dataset* dataset, Tree* tree) {}
  virtual void SampleLeafParameters(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, Tree* tree) {}
  virtual void SampleGlobalParameters(Dataset* dataset, TreeEnsemble* tree_ensemble, std::set<std::string> update_params) {}
  virtual double GetGlobalParameter(std::string param_name) {}
  virtual void SetGlobalParameter(std::string param_name, double param_value) {}
};

struct XBARTGaussianRegressionSuffStat {
  data_size_t sample_size_;
  double outcome_sum_;
  double outcome_sum_sq_;
};

class XBARTGaussianRegressionModel : public Model {
 public:
  XBARTGaussianRegressionModel();
  XBARTGaussianRegressionModel(const Config& config);
  ~XBARTGaussianRegressionModel() {}

  /*! \brief Initialize hyperparameters for the global variance parameters (using the training data) */
  void InitializeGlobalParameters(Dataset* dataset);
  
  /*! \brief Sample a tree via the "grow from root" algorithm (He and Hahn 2021) */
  void SampleTree(Dataset* dataset, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                  SampleNodeMapper* sample_node_mapper, int tree_num);
  
  /*! \brief Sample leaf parameters, which are normal conditional on the data, tree structure, and variance parameters */
  void SampleLeafParameters(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, Tree* tree);
  
  /*! \brief Compute sufficient statistics for node leaf_id */
  XBARTGaussianRegressionSuffStat LeafSuffStat(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, node_t leaf_id);
  
  /*! \brief Sample residual variance and leaf node variance parameters, both IG */
  void SampleGlobalParameters(Dataset* dataset, TreeEnsemble* tree_ensemble, std::set<std::string> update_params);
  
  /*! \brief A way of calling the recursive "grow from root" procedure from a given 
   *  leaf node without actual recursive calls to a function. 
   *  The full grow-from-root procedure is initiated by calling `GrowFromLeaf(tree, 0);`
   */
  void SampleSplitRule(Dataset* dataset, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                       node_t leaf_node, data_size_t node_begin, data_size_t node_end, std::deque<node_t>& split_queue, 
                       SampleNodeMapper* sample_node_mapper, int tree_num);
  
  /*! \brief Compute the marginal log-likelihood of a proposed split, using sufficient statistics 
   *  accumulated for the potential left and right nodes */
  double SplitLogMarginalLikelihood(const XBARTGaussianRegressionSuffStat& left_stat, 
                                    const XBARTGaussianRegressionSuffStat& right_stat);

  /*! \brief Compute the marginal likelihood of a proposed split, using sufficient statistics 
   *  accumulated for the potential left and right nodes */
  double SplitMarginalLikelihood(const XBARTGaussianRegressionSuffStat& left_stat, 
                                 const XBARTGaussianRegressionSuffStat& right_stat);
  
  /*! \brief Compute the marginal log-likelihood of not splitting a node */
  double NoSplitLogMarginalLikelihood(const XBARTGaussianRegressionSuffStat& node_stat, 
                                      int node_depth, data_size_t num_split_candidates);
  
  /*! \brief Compute the marginal llikelihood of not splitting a node */
  double NoSplitMarginalLikelihood(const XBARTGaussianRegressionSuffStat& node_stat, 
                                   int node_depth, data_size_t num_split_candidates);
  
  /*! \brief Determine the number of possible cutpoints for a given node of a given tree */
  void Cutpoints(Dataset* dataset, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                 node_t leaf_node, data_size_t node_begin, data_size_t node_end, 
                 std::vector<double>& log_cutpoint_evaluations, 
                 std::vector<int>& cutpoint_feature, 
                 std::vector<double>& cutpoint_values, 
                 std::vector<FeatureType>& cutpoint_feature_types, 
                 data_size_t& valid_cutpoint_count);
  
  /*! \brief Add a split to the model by growing the tree and modifying the training tracker */
  void AddSplitToModel(Dataset* dataset, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                       FeatureType feature_type, node_t leaf_node, data_size_t node_begin, data_size_t node_end, 
                       int feature_split, double split_value, std::deque<node_t>& split_queue, 
                       SampleNodeMapper* sample_node_mapper, int tree_num);

  /*! \brief Compute sufficient statistics for a given tree node */
  XBARTGaussianRegressionSuffStat ComputeNodeSuffStat(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                                                      data_size_t node_begin, data_size_t node_end, int feature_idx);

  /*! \brief Subtract the second sufficient statistic's entries from the first's sufficient statistics entries */
  XBARTGaussianRegressionSuffStat SubtractSuffStat(const XBARTGaussianRegressionSuffStat& first_node_suff_stat, 
                                                   const XBARTGaussianRegressionSuffStat& second_node_suff_stat);

  /*! \brief Accumulate sufficient statistics based on a proposed split to a node, 
   *  by default assumes we are accumulating statistics for the new left node, which 
   *  can be changed by is_left */
  void AccumulateSplitRule(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker,
                           XBARTGaussianRegressionSuffStat& suff_stat, 
                           int split_col, double split_value, data_size_t node_begin, 
                           data_size_t node_end, bool is_left = true);

  /*! \brief Accumulate sufficient statistics based on a proposed split to a node, 
   *  by default assumes we are accumulating statistics for the new left node, which 
   *  can be changed by is_left */
  void AccumulateSplitRule(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                           XBARTGaussianRegressionSuffStat& suff_stat, 
                           int split_col, std::vector<std::uint32_t> const& categorical_indices, 
                           data_size_t node_begin, data_size_t node_end, bool is_left = true);

  /*! \brief Reset sufficient statistic to specific values (default 0) */
  void ResetSuffStat(XBARTGaussianRegressionSuffStat& suff_stat, data_size_t sample_size = 0, double outcome_sum = 0., double outcome_sum_sq = 0.);
  
  /*! \brief Add node to the node tracking table */
  inline void AddNode(node_t node_id, data_size_t node_begin, data_size_t node_end) {
    node_index_map_.insert({node_id, std::make_pair(node_begin, node_end)});
  }

  /*! \brief Compute the posterior mean of a leaf node, given the other model parameters and the samples that reach that node */
  inline double LeafPosteriorMean(XBARTGaussianRegressionSuffStat& leaf_suff_stat) {
    return leaf_suff_stat.outcome_sum_ / sigma_sq_ / ((1./tau_) + (leaf_suff_stat.sample_size_/sigma_sq_));
  }

  /*! \brief Compute the posterior standard deviation of a leaf node, given the other model parameters and the samples that reach that node */
  inline double LeafPosteriorStddev(XBARTGaussianRegressionSuffStat& leaf_suff_stat) {
    return std::sqrt(1. / ((1./tau_) + (leaf_suff_stat.sample_size_/sigma_sq_)));
  }

  /*! \brief Compute the posterior shape parameter for tau, the variance terms in the prior on leaf node parameters */
  inline double TauPosteriorShape(TreeEnsemble* tree_ensemble) {
    int num_leaves = 0;
    for (int i = 0; i < config_.num_trees; i++) {
      for (auto& leaf: tree_ensemble->GetTree(i)->GetLeaves()) {
        num_leaves += 1;
      }
    }
    return (a_tau_ + num_leaves)/2.;
  }

  /*! \brief Compute the posterior scale parameter for tau, the variance terms in the prior on leaf node parameters */
  inline double TauPosteriorScale(TreeEnsemble* tree_ensemble) {
    double sum_sq_leaf_vals = 0;
    for (int i = 0; i < config_.num_trees; i++) {
      for (auto& leaf: tree_ensemble->GetTree(i)->GetLeaves()) {
        // sum_sq_leaf_vals += std::pow((*tree_ensemble->GetTree(i))[leaf].LeafValue(), 2.);
        sum_sq_leaf_vals += std::pow((*tree_ensemble->GetTree(i)).LeafValue(leaf), 2.);
      }
    }
    return (b_tau_ + sum_sq_leaf_vals)/2.;
  }

  /*! \brief Compute the posterior shape parameter for sigma^2, the global residual error variance */
  inline double SigmaPosteriorShape(Dataset* dataset) {
    data_size_t n = dataset->NumObservations();
    return (a_sigma_ + n)/2.;
  }

  /*! \brief Compute the posterior scale parameter for sigma^2, the global residual error variance */
  inline double SigmaPosteriorScale(Dataset* dataset) {
    data_size_t n = dataset->NumObservations();
    double sum_sq_resid = 0.;
    for (data_size_t i = 0; i < n; i++) {
      // TODO: generalize to vector-valued residuals
      sum_sq_resid += std::pow(dataset->ResidualValue(i), 2);
    }
    return (b_sigma_ + sum_sq_resid)/2.;
  }

  inline double GetGlobalParameter(std::string param_name) {
    if (param_name == "tau") {
      return tau_;
    } else if (param_name == "sigma_sq") {
      return sigma_sq_;
    } else if (param_name == "ybar_offset") {
      return ybar_offset_;
    } else if (param_name == "sd_scale") {
      return sd_scale_;
    }
  }

  inline void SetGlobalParameter(std::string param_name, double param_value) {
    if (param_name == "tau") {
      tau_ = param_value;
    } else if (param_name == "sigma_sq") {
      sigma_sq_ = param_value;
    } else if (param_name == "ybar_offset") {
      ybar_offset_ = param_value;
    } else if (param_name == "sd_scale") {
      sd_scale_ = param_value;
    }
  }

  inline void NodeIndexMapReset(data_size_t n) {
    node_index_map_.clear();
    AddNode(0, 0, n);
  }

  /*! @brief Determine the split condition for an underordered categorical feature */
  inline void DetermineCategoricalSplit(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, int& num_categories, std::vector<std::uint32_t>& categories, int32_t split_feature, double split_value, data_size_t node_begin, data_size_t node_end) {
    if (dataset->GetFeatureType(split_feature) != kUnorderedCategorical) {
      Log::Fatal("Feature must be unordered categorical");
    }
    data_size_t stride_length;
    double category_value;
    num_categories = 0;
    categories.clear();
    std::set<std::uint32_t> category_set;
    data_size_t i = node_begin;
    data_size_t sort_idx;
    int bin_idx;
    while (i < node_end) {
      num_categories++;
      sort_idx = sorted_node_sample_tracker->SortIndex(i, split_feature);
      category_value = dataset->CovariateValue(sort_idx, split_feature);
      if (category_value <= split_value) {
        category_set.insert(category_value);
      }
      // TODO
      // stride_length = node_cutpoint_tracker.node_stride_length_[bin_idx];
      i += 1;
      bin_idx++;
    }    
  }

  /*! @brief Partition a leaf node based on a categorical split */
  inline void PartitionLeaf(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, node_t leaf_node, 
                            data_size_t leaf_start_idx, data_size_t num_leaf_elements, int split_col, std::vector<std::uint32_t>& categories) {
    sorted_node_sample_tracker->PartitionNode(dataset, leaf_node, split_col, categories);
  }

  /*! @brief Partition a leaf node based on a numeric split */
  inline void PartitionLeaf(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, node_t leaf_node, 
                            data_size_t leaf_start_idx, data_size_t num_leaf_elements, int split_col, double split_value) {
    sorted_node_sample_tracker->PartitionNode(dataset, leaf_node, split_col, split_value);
  }

 private:
  Config config_;
  double alpha_;
  double beta_;
  double a_sigma_;
  double a_tau_;
  double b_sigma_;
  double b_tau_;
  double sigma_sq_;
  double tau_;
  double ybar_offset_;
  double sd_scale_;
  std::unordered_map<node_t, std::pair<data_size_t, data_size_t>> node_index_map_;
  std::mt19937 gen;
  std::normal_distribution<double> leaf_node_dist_;
  std::gamma_distribution<double> residual_variance_dist_;
  std::gamma_distribution<double> leaf_node_variance_dist_;
  std::unique_ptr<CutpointGridContainer> cutpoint_grid_container;
};

struct BARTGaussianRegressionSuffStat {
  data_size_t sample_size_;
  double outcome_sum_;
  double outcome_sum_sq_;
};

class BARTGaussianRegressionModel : public Model {
 public:
  BARTGaussianRegressionModel();
  BARTGaussianRegressionModel(const Config& config);
  ~BARTGaussianRegressionModel() {}

  /*! \brief Initialize hyperparameters for the global variance parameters (using the training data) */
  void InitializeGlobalParameters(Dataset* dataset);
  
  /*! \brief Sample the tree */
  void SampleTree(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, SampleNodeMapper* sample_node_mapper, int tree_num);
  
  /*! \brief Perform one MCMC grow-prune step */
  void MCMCTreeStep(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, int tree_num);
  void GrowMCMC(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, bool& accept, int tree_num);
  void PruneMCMC(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, bool& accept, int tree_num);
  
  /*! \brief Compute sufficient statistics for node indexed by node_id in tree */
  BARTGaussianRegressionSuffStat ComputeNodeSufficientStatistics(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, int node_id, int tree_num);
  
  /*! \brief Sample leaf parameters, which are normal conditional on the data, tree structure, and variance parameters */
  void SampleLeafParameters(Dataset* dataset, Tree* tree);
  
  /*! \brief Sample residual variance and leaf node variance parameters, both IG */
  void SampleGlobalParameters(Dataset* dataset, TreeEnsemble* tree_ensemble, std::set<std::string> update_params);
  
  /*! \brief Compute the marginal log-likelihood of a proposed split, using sufficient statistics 
   *  accumulated for the potential left and right nodes */
  double SplitLogMarginalLikelihood(const BARTGaussianRegressionSuffStat& left_stat, 
                                    const BARTGaussianRegressionSuffStat& right_stat);
  
  /*! \brief Compute the marginal likelihood of a proposed split, using sufficient statistics 
   *  accumulated for the potential left and right nodes */
  double SplitMarginalLikelihood(const BARTGaussianRegressionSuffStat& left_stat, 
                                 const BARTGaussianRegressionSuffStat& right_stat);
  
  /*! \brief Compute the marginal log-likelihood of not splitting a node */
  double NoSplitLogMarginalLikelihood(const BARTGaussianRegressionSuffStat& node_stat);
  
  /*! \brief Compute the marginal likelihood of not splitting a node */
  double NoSplitMarginalLikelihood(const BARTGaussianRegressionSuffStat& node_stat);
  
  /*! \brief Add a split to the model by growing the tree and modifying the training tracker */
  void AddSplitToModel(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, node_t leaf_node, 
                       int feature_split, double split_value, BARTGaussianRegressionSuffStat& node_suff_stat, 
                       BARTGaussianRegressionSuffStat& left_suff_stat, BARTGaussianRegressionSuffStat& right_suff_stat, int tree_num);
  
  /*! \brief Remove a split from the model by growing the tree and modifying the training tracker */
  void RemoveSplitFromModel(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, node_t leaf_node, node_t left_node, 
                            node_t right_node, int feature_split, double split_value, BARTGaussianRegressionSuffStat& node_suff_stat, 
                            BARTGaussianRegressionSuffStat& left_suff_stat, BARTGaussianRegressionSuffStat& right_suff_stat, int tree_num);

  /*! \brief Compute sufficient statistic for a leaf */
  void LeafSuffStat(Dataset* dataset, Tree* tree, SampleNodeMapper* sample_node_mapper, 
                    int tree_num, int leaf_split, BARTGaussianRegressionSuffStat& node_suff_stat);
  
  /*! \brief Convert a split node to a leaf */
  void ConvertSplitToLeaf(Dataset* dataset, Tree* tree, node_t leaf_node, 
                          data_size_t node_begin, data_size_t node_end, 
                          int feature_split, double split_value, std::deque<node_t> split_queue, 
                          SampleNodeMapper* sample_node_mapper, int tree_num);
  
  /*! \brief Determine whether newly proposed left and right nodes have non-constant features (i.e. are able to split) */
  bool NodesNonConstantAfterSplit(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, 
                                  int leaf_split, int feature_split, double split_value, int tree_num);
  
  /*! \brief Determine whether a node has non-constant features */
  bool NodeNonConstant(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, int node_id, int tree_num);

  /*! \brief Compute sufficient statistics for a chosen leaf node and its new left and right nodes */
  void ComputeLeafNodeSuffStats(Dataset* dataset, Tree* tree, SampleNodeMapper* sample_node_mapper, 
                                int tree_num, int leaf_split, BARTGaussianRegressionSuffStat& node_suff_stat, 
                                BARTGaussianRegressionSuffStat& left_suff_stat, BARTGaussianRegressionSuffStat& right_suff_stat, 
                                int feature_split, double split_value);

  /*! \brief Sufficient statistics for the left and right nodes produces by a proposed split */
  void ComputeSplitSuffStats(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, 
                             int leaf_split, BARTGaussianRegressionSuffStat& left_suff_stat, 
                             BARTGaussianRegressionSuffStat& right_suff_stat, 
                             int feature_split, double split_value, int tree_num);
  
  /*! \brief Determine the minimum and maximum values available to split for a given feature in a given leaf */
  void VarSplitRange(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, 
                     int leaf_split, int feature_split, double& var_min, double& var_max, int tree_num);

  /*! \brief Reset sufficient statistic to specific values (default 0) */
  void ResetSuffStat(BARTGaussianRegressionSuffStat& suff_stat, data_size_t sample_size = 0, double outcome_sum = 0., double outcome_sum_sq = 0.);
  
  /*! \brief Compute the posterior mean of a leaf node, given the other model parameters and the samples that reach that node */
  inline double LeafPosteriorMean(BARTGaussianRegressionSuffStat& leaf_suff_stat) {
    data_size_t n_leaf = leaf_suff_stat.sample_size_;
    double y_bar = leaf_suff_stat.outcome_sum_ / ((double) n_leaf);
    double prior_contrib = (sigma_sq_/(sigma_sq_ + n_leaf*mu_sigma_))*mu_mean_;
    double data_contrib = ((n_leaf*mu_sigma_)/(sigma_sq_ + n_leaf*mu_sigma_))*y_bar;
    return prior_contrib + data_contrib;
  }

  /*! \brief Compute the posterior standard deviation of a leaf node, given the other model parameters and the samples that reach that node */
  inline double LeafPosteriorStddev(BARTGaussianRegressionSuffStat& leaf_suff_stat) {
    data_size_t n_leaf = leaf_suff_stat.sample_size_;
    return ((sigma_sq_*mu_sigma_)/(sigma_sq_ + n_leaf*mu_sigma_));
  }

  /*! \brief Compute the posterior shape parameter for sigma^2, the global residual error variance */
  inline double SigmaPosteriorShape(Dataset* dataset) {
    data_size_t n = dataset->NumObservations();
    return (nu_/2.0) + n;
  }
  
  /*! \brief Compute the posterior scale parameter for sigma^2, the global residual error variance */
  inline double SigmaPosteriorScale(Dataset* dataset) {
    data_size_t n = dataset->NumObservations();
    double nu_lambda_ = nu_*lambda_;
    double sum_sq_resid = 0.;
    for (data_size_t i = 0; i < n; i++) {
      sum_sq_resid += std::pow(dataset->ResidualValue(i), 2);
    }
    return (nu_lambda_/2.0) + sum_sq_resid;
  }

  inline double GetGlobalParameter(std::string param_name) {
    if (param_name == "sigma_sq") {
      return sigma_sq_;
    } else if (param_name == "ybar_offset") {
      return ybar_offset_;
    } else if (param_name == "sd_scale") {
      return sd_scale_;
    }
  }

  inline void SetGlobalParameter(std::string param_name, double param_value) {
    if (param_name == "sigma_sq") {
      sigma_sq_ = param_value;
    } else if (param_name == "ybar_offset") {
      ybar_offset_ = param_value;
    } else if (param_name == "sd_scale") {
      sd_scale_ = param_value;
    }
  }

 private:
  Config config_;
  double alpha_;
  double beta_;
  double nu_;
  double lambda_;
  double mu_mean_;
  double mu_sigma_;
  double sigma_sq_;
  double ybar_offset_;
  double sd_scale_;
  std::vector<BARTGaussianRegressionSuffStat> node_suff_stats_;
  std::mt19937 gen;
  std::normal_distribution<double> leaf_node_dist_;
  std::gamma_distribution<double> residual_variance_dist_;
};

} // namespace StochTree

#endif // STOCHTREE_MODEL_H_
