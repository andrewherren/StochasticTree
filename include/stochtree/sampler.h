/*!
 * Copyright (c) 2024 stochtree authors. All rights reserved.
 */
#ifndef STOCHTREE_SAMPLER_H_
#define STOCHTREE_SAMPLER_H_

#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/prior.h>

#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace StochTree {

template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
void VarSplitRange(ForestDatasetType* data, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int leaf_split, int feature_split, double& var_min, double& var_max, int tree_num) {
  data_size_t n = data->covariates.rows();
  var_min = std::numeric_limits<double>::max();
  var_max = std::numeric_limits<double>::min();
  double feature_value;
  auto tree_node_tracker = node_tracker->GetFeaturePartition(tree_num);
  data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
  data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);
  data_size_t idx;
  auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
  auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    idx = *i;
    feature_value = data->covariates(idx, feature_split);
    if (feature_value < var_min) {
      var_min = feature_value;
    } else if (feature_value > var_max) {
      var_max = feature_value;
    }
  }
}

template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
void AddSplitToModel(ForestDatasetType* data, Tree* tree, UnsortedNodeSampleTracker* node_tracker, SampleNodeMapper* sample_node_mapper, int leaf_node, int feature_split, double split_value, int tree_num) {
  // Use zeros as a "temporary" leaf values since we draw leaf parameters after tree sampling is complete
  int basis_dim = 1;
  if (basis_dim > 1) {
    std::vector<double> temp_leaf_values(basis_dim, 0.);
    tree->ExpandNode(leaf_node, feature_split, split_value, true, temp_leaf_values, temp_leaf_values);
  } else {
    CHECK_EQ(basis_dim, 1);
    double temp_leaf_value = 0.;
    tree->ExpandNode(leaf_node, feature_split, split_value, true, temp_leaf_value, temp_leaf_value);
  }
  int left_node = tree->LeftChild(leaf_node);
  int right_node = tree->RightChild(leaf_node);

  // Update the UnsortedNodeSampleTracker
  node_tracker->PartitionTreeNode(data->covariates, tree_num, leaf_node, left_node, right_node, feature_split, split_value);

  // Update the SampleNodeMapper
  node_tracker->UpdateObservationMapping(tree, tree_num, sample_node_mapper);
}

template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
void RemoveSplitFromModel(ForestDatasetType* data, Tree* tree, UnsortedNodeSampleTracker* node_tracker, SampleNodeMapper* sample_node_mapper, int leaf_node, int left_node, int right_node, int feature_split, double split_value, int tree_num) {
  // Use zeros as a "temporary" leaf values since we draw leaf parameters after tree sampling is complete
  int basis_dim = 1;
  if (basis_dim > 1) {
    std::vector<double> temp_leaf_values(basis_dim, 0.);
    tree->ChangeToLeaf(leaf_node, temp_leaf_values);
  } else {
    CHECK_EQ(basis_dim, 1);
    double temp_leaf_value = 0.;
    tree->ChangeToLeaf(leaf_node, temp_leaf_value);
  }

  // Update the UnsortedNodeSampleTracker
  node_tracker->PruneTreeNodeToLeaf(tree_num, leaf_node);

  // Update the SampleNodeMapper
  node_tracker->UpdateObservationMapping(tree, tree_num, sample_node_mapper);
}

template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
double SplitLogMarginalLikelihood(ForestLeafSuffStatType& left_stat, ForestLeafSuffStatType& right_stat, ForestLeafPriorType& leaf_prior, double global_variance) {
  double sigma_sq = global_variance;
  double tau = leaf_prior.GetPriorScale();
  
  // Compute the log marginal likelihood for the left node
  double left_n = static_cast<double>(left_stat.n);
  double left_sum_y = left_stat.sum_y;
  double left_sum_y_squared = left_stat.sum_y_squared;
  double left_log_ml = (
    -(left_n*0.5)*std::log(2*M_PI) - (left_n)*std::log(std::sqrt(sigma_sq)) + 
    (0.5)*std::log(sigma_sq/(sigma_sq + tau*left_n)) - (left_sum_y_squared/(2.0*sigma_sq)) + 
    ((tau*std::pow(left_sum_y, 2.0))/(2*sigma_sq*(sigma_sq + tau*left_n)))
  );

  // Compute the log marginal likelihood for the right node
  double right_n = static_cast<double>(right_stat.n);
  double right_sum_y = right_stat.sum_y;
  double right_sum_y_squared = right_stat.sum_y_squared;
  double right_log_ml = (
    -(right_n*0.5)*std::log(2*M_PI) - (right_n)*std::log(std::sqrt(sigma_sq)) + 
    (0.5)*std::log(sigma_sq/(sigma_sq + tau*right_n)) - (right_sum_y_squared/(2.0*sigma_sq)) + 
    ((tau*std::pow(right_sum_y, 2.0))/(2*sigma_sq*(sigma_sq + tau*right_n)))
  );

  // Return the combined log marginal likelihood
  return left_log_ml + right_log_ml;
}

template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
double NoSplitLogMarginalLikelihood(ForestLeafSuffStatType& suff_stat, ForestLeafPriorType& leaf_prior, double global_variance) {
  double sigma_sq = global_variance;
  double tau = leaf_prior.GetPriorScale();
  double n = static_cast<double>(suff_stat.n);
  double sum_y = suff_stat.sum_y;
  double sum_y_squared = suff_stat.sum_y_squared;
  double log_ml = (
    -(n*0.5)*std::log(2*M_PI) - (n)*std::log(std::sqrt(sigma_sq)) + 
    (0.5)*std::log(sigma_sq/(sigma_sq + tau*n)) - (sum_y_squared/(2.0*sigma_sq)) + 
    ((tau*std::pow(sum_y, 2.0))/(2*sigma_sq*(sigma_sq + tau*n)))
  );

  return log_ml;
}

template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
bool NodeNonConstant(ForestDatasetType* data, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int node_id, int tree_num) {
  int p = data->covariates.cols();
  double outcome_value;
  double feature_value;
  double split_feature_value;
  double var_max;
  double var_min;
  auto tree_node_tracker = node_tracker->GetFeaturePartition(tree_num);
  data_size_t node_begin = tree_node_tracker->NodeBegin(node_id);
  data_size_t node_end = tree_node_tracker->NodeEnd(node_id);
  data_size_t idx;

  for (int j = 0; j < p; j++) {
    var_max = std::numeric_limits<double>::min();
    var_min = std::numeric_limits<double>::max();
    auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
    auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
    for (auto i = node_begin_iter; i != node_end_iter; i++) {
      idx = *i;
      feature_value = data->covariates(idx, j);
      if (var_max < feature_value) {
        var_max = feature_value;
      } else if (var_min > feature_value) {
        var_max = feature_value;
      }
    }
    if (var_max > var_min) {
      return true;
    }
  }
  return false;
}

class TreeSampler {
 public:
  TreeSampler() {}
  virtual ~TreeSampler() = default;
  virtual void AssignAllSamplesToRoot(int tree_num) {}
  virtual data_size_t GetNodeId(int observation_num, int tree_num) {}
};

class MCMCTreeSampler : public TreeSampler {
 public:
  MCMCTreeSampler() {}
  ~MCMCTreeSampler() {}
  
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  void Initialize(ForestDatasetType* data, int num_trees, int num_observations, std::vector<FeatureType>& feature_types) {
    sample_node_mapper_ = std::make_unique<SampleNodeMapper>(num_trees, num_observations);
    unsorted_node_sample_tracker_ = std::make_unique<UnsortedNodeSampleTracker>(num_observations, num_trees);
    for (int i = 0; i < num_trees; i++) {
      AssignAllSamplesToRoot(i);
    }
  }
  
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  void SampleTree(Tree* tree, UnivariateResidual* residual, ForestDatasetType* data, ForestLeafPriorType* leaf_prior, TreePrior* tree_prior, double outcome_variance, std::mt19937& gen, int32_t tree_num, int32_t cutpoint_grid_size, std::vector<FeatureType>& feature_types) {
    // Determine whether it is possible to grow any of the leaves
    bool grow_possible = false;
    std::vector<int> leaves = tree->GetLeaves();
    for (auto& leaf: leaves) {
      if (unsorted_node_sample_tracker_->NodeSize(tree_num, leaf) > 2 * tree_prior->GetMinSamplesLeaf()) {
        grow_possible = true;
        break;
      }
    }

    // Determine whether it is possible to prune the tree
    bool prune_possible = false;
    if (tree->NumValidNodes() > 1) {
      prune_possible = true;
    }

    // Determine the relative probability of grow vs prune (0 = grow, 1 = prune)
    double prob_grow;
    std::vector<double> step_probs(2);
    if (grow_possible && prune_possible) {
      step_probs = {0.5, 0.5};
      prob_grow = 0.5;
    } else if (!grow_possible && prune_possible) {
      step_probs = {0.0, 1.0};
      prob_grow = 0.0;
    } else if (grow_possible && !prune_possible) {
      step_probs = {1.0, 0.0};
      prob_grow = 1.0;
    } else {
      Log::Fatal("In this tree, neither grow nor prune is possible");
    }
    std::discrete_distribution<> step_dist(step_probs.begin(), step_probs.end());

    // Draw a split rule at random
    data_size_t step_chosen = step_dist(gen);
    bool accept;
    
    if (step_chosen == 0) {
      GrowMCMC<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(tree, residual, data, leaf_prior, tree_prior, outcome_variance, gen, tree_num, prob_grow);
    } else {
      PruneMCMC<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(tree, residual, data, leaf_prior, tree_prior, outcome_variance, gen, tree_num);
    }
  }

  void AssignAllSamplesToRoot(int tree_num);
  data_size_t GetNodeId(int observation_num, int tree_num);
  
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  void Reset(TreeEnsembleContainer* container, ForestDatasetType* data, std::vector<FeatureType>& feature_types, int tree_num, int sample_num, int prev_sample_num) {
    CHECK_GE(prev_sample_num, 0);
    Tree* prev_tree = container->GetEnsemble(prev_sample_num)->GetTree(tree_num);
    container->GetEnsemble(sample_num)->ResetTree(tree_num);
    container->GetEnsemble(sample_num)->CloneFromExistingTree(tree_num, prev_tree);
  }

  UnsortedNodeSampleTracker* GetNodeSampleTracker() {return unsorted_node_sample_tracker_.get();}
 
 private:
  /*! \brief Mapper from observations to leaf node indices for every tree in a forest */
  std::unique_ptr<SampleNodeMapper> sample_node_mapper_;
  /*! \brief Data structure tracking / updating observations available in each node for every tree in a forest */
  std::unique_ptr<UnsortedNodeSampleTracker> unsorted_node_sample_tracker_;

  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  void GrowMCMC(Tree* tree, UnivariateResidual* residual, ForestDatasetType* data, ForestLeafPriorType* leaf_prior, TreePrior* tree_prior, double outcome_variance, std::mt19937& gen, int32_t tree_num, double prob_grow_old) {
    // Extract dataset information
    data_size_t n = data->covariates.rows();
    int basis_dim = 1;

    // Choose a leaf node at random
    int num_leaves = tree->NumLeaves();
    std::vector<int> leaves = tree->GetLeaves();
    std::vector<double> leaf_weights(num_leaves);
    std::fill(leaf_weights.begin(), leaf_weights.end(), 1.0/num_leaves);
    std::discrete_distribution<> leaf_dist(leaf_weights.begin(), leaf_weights.end());
    int leaf_chosen = leaves[leaf_dist(gen)];
    int leaf_depth = tree->GetDepth(leaf_chosen);

    // Select a split variable at random
    int p = data->covariates.cols();
    std::vector<double> var_weights(p);
    std::fill(var_weights.begin(), var_weights.end(), 1.0/p);
    std::discrete_distribution<> var_dist(var_weights.begin(), var_weights.end());
    int var_chosen = var_dist(gen);

    // Determine the range of possible cutpoints
    double var_min, var_max;
    VarSplitRange<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(data, tree, unsorted_node_sample_tracker_.get(), leaf_chosen, var_chosen, var_min, var_max, tree_num);
    if (var_max <= var_min) {
      return;
    }
    // Split based on var_min to var_max in a given node
    std::uniform_real_distribution<double> split_point_dist(var_min, var_max);
    double split_point_chosen = split_point_dist(gen);

    // Compute the marginal likelihood of split and no split, given the leaf prior
    std::tuple<double, double, int32_t, int32_t> split_eval = EvaluateSplitAndNoSplit<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(tree, residual, data, leaf_prior, tree_prior, outcome_variance, tree_num, leaf_chosen, var_chosen, split_point_chosen);
    double split_log_marginal_likelihood = std::get<0>(split_eval);
    double no_split_log_marginal_likelihood = std::get<1>(split_eval);
    int32_t left_n = std::get<2>(split_eval);
    int32_t right_n = std::get<3>(split_eval);
    
    // Determine probability of growing the split node and its two new left and right nodes
    double pg = tree_prior->GetAlpha() * std::pow(1+leaf_depth, -tree_prior->GetBeta());
    double pgl = tree_prior->GetAlpha() * std::pow(1+leaf_depth+1, -tree_prior->GetBeta());
    double pgr = tree_prior->GetAlpha() * std::pow(1+leaf_depth+1, -tree_prior->GetBeta());

    // Determine whether a "grow" move is possible from the newly formed tree
    // in order to compute the probability of choosing "prune" from the new tree
    // (which is always possible by construction)
    bool non_constant = NodesNonConstantAfterSplit<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(data, tree, leaf_chosen, var_chosen, split_point_chosen, tree_num);
//    bool right_non_constant = NodesNonConstantAfterSplit<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(data, tree, leaf_chosen, var_chosen, split_point_chosen, tree_num, false);
    bool min_samples_left_check = left_n >= 2*tree_prior->GetMinSamplesLeaf();
    bool min_samples_right_check = right_n >= 2*tree_prior->GetMinSamplesLeaf();
    double prob_prune_new;
    if (non_constant && (min_samples_left_check || min_samples_right_check)) {
      prob_prune_new = 0.5;
    } else {
      prob_prune_new = 1.0;
    }

    // Determine the number of leaves in the current tree and leaf parents in the proposed tree
    int num_leaf_parents = tree->NumLeafParents();
    double p_leaf = 1/num_leaves;
    double p_leaf_parent = 1/(num_leaf_parents+1);

    // Compute the final MH ratio
    double log_mh_ratio = (
      std::log(pg) + std::log(1-pgl) + std::log(1-pgr) - std::log(1-pg) + std::log(prob_prune_new) +
      std::log(p_leaf_parent) - std::log(prob_grow_old) - std::log(p_leaf) + no_split_log_marginal_likelihood - split_log_marginal_likelihood
    );
    // Threshold at 0
    if (log_mh_ratio > 1) {
      log_mh_ratio = 1;
    }

    // Draw a uniform random variable and accept/reject the proposal on this basis
    bool accept;
    std::uniform_real_distribution<double> mh_accept(0.0, 1.0);
    double log_acceptance_prob = std::log(mh_accept(gen));
    if (log_acceptance_prob <= log_mh_ratio) {
      accept = true;
      AddSplitToModel<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(data, tree, unsorted_node_sample_tracker_.get(), sample_node_mapper_.get(), leaf_chosen, var_chosen, split_point_chosen, tree_num);
    } else {
      accept = false;
    }
  }
  
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  void PruneMCMC(Tree* tree, UnivariateResidual* residual, ForestDatasetType* data, ForestLeafPriorType* leaf_prior, TreePrior* tree_prior, double outcome_variance, std::mt19937& gen, int32_t tree_num) {
    // Choose a "leaf parent" node at random
    int num_leaves = tree->NumLeaves();
    int num_leaf_parents = tree->NumLeafParents();
    std::vector<int> leaf_parents = tree->GetLeafParents();
    std::vector<double> leaf_parent_weights(num_leaf_parents);
    std::fill(leaf_parent_weights.begin(), leaf_parent_weights.end(), 1.0/num_leaf_parents);
    std::discrete_distribution<> leaf_parent_dist(leaf_parent_weights.begin(), leaf_parent_weights.end());
    int leaf_parent_chosen = leaf_parents[leaf_parent_dist(gen)];
    int leaf_parent_depth = tree->GetDepth(leaf_parent_chosen);
    int left_node = tree->LeftChild(leaf_parent_chosen);
    int right_node = tree->RightChild(leaf_parent_chosen);
    int feature_split = tree->SplitIndex(leaf_parent_chosen);
    double split_value = tree->Threshold(leaf_parent_chosen);

    // Compute the marginal likelihood for the leaf parent and its left and right nodes
    std::tuple<double, double, int32_t, int32_t> split_eval = EvaluateSplitAndNoSplit<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(tree, residual, data, leaf_prior, tree_prior, outcome_variance, tree_num, leaf_parent_chosen, left_node, right_node);
    double split_log_marginal_likelihood = std::get<0>(split_eval);
    double no_split_log_marginal_likelihood = std::get<1>(split_eval);
    int32_t left_n = std::get<2>(split_eval);
    int32_t right_n = std::get<3>(split_eval);
    
    // Determine probability of growing the split node and its two new left and right nodes
    double pg = tree_prior->GetAlpha() * std::pow(1+leaf_parent_depth, -tree_prior->GetBeta());
    double pgl = tree_prior->GetAlpha() * std::pow(1+leaf_parent_depth+1, -tree_prior->GetBeta());
    double pgr = tree_prior->GetAlpha() * std::pow(1+leaf_parent_depth+1, -tree_prior->GetBeta());

    // Determine whether a "prune" move is possible from the new tree,
    // in order to compute the probability of choosing "grow" from the new tree
    // (which is always possible by construction)
    bool non_root_tree = tree->NumNodes() > 1;
    double prob_grow_new;
    if (non_root_tree) {
      prob_grow_new = 0.5;
    } else {
      prob_grow_new = 1.0;
    }

    // Determine whether a "grow" move was possible from the old tree,
    // in order to compute the probability of choosing "prune" from the old tree
    bool non_constant_left = NodeNonConstant<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(data, tree, unsorted_node_sample_tracker_.get(), left_node, tree_num);
    bool non_constant_right = NodeNonConstant<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(data, tree, unsorted_node_sample_tracker_.get(), right_node, tree_num);
    double prob_prune_old;
    if (non_constant_left && non_constant_right) {
      prob_prune_old = 0.5;
    } else {
      prob_prune_old = 1.0;
    }

    // Determine the number of leaves in the current tree and leaf parents in the proposed tree
    double p_leaf = 1/(num_leaves-1);
    double p_leaf_parent = 1/(num_leaf_parents);

    // Compute the final MH ratio
    double log_mh_ratio = (
      std::log(1-pg) - std::log(pg) - std::log(1-pgl) - std::log(1-pgr) + std::log(prob_prune_old) +
      std::log(p_leaf) - std::log(prob_grow_new) - std::log(p_leaf_parent) + no_split_log_marginal_likelihood - split_log_marginal_likelihood
    );
    // Threshold at 0
    if (log_mh_ratio > 0) {
      log_mh_ratio = 0;
    }

    // Draw a uniform random variable and accept/reject the proposal on this basis
    bool accept;
    std::uniform_real_distribution<double> mh_accept(0.0, 1.0);
    double log_acceptance_prob = std::log(mh_accept(gen));
    if (log_acceptance_prob <= log_mh_ratio) {
      accept = true;
      RemoveSplitFromModel<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(data, tree, unsorted_node_sample_tracker_.get(), sample_node_mapper_.get(), leaf_parent_chosen, left_node, right_node, feature_split, split_value, tree_num);
    } else {
      accept = false;
    }
  }
  
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  std::tuple<double, double, int32_t, int32_t> EvaluateSplitAndNoSplit(Tree* tree, UnivariateResidual* residual, ForestDatasetType* data, ForestLeafPriorType* leaf_prior, TreePrior* tree_prior, double outcome_variance, int32_t tree_num, int leaf_split, int feature_split, double split_value) {
    // Unpack shifted iterators to observations in a given node
    auto tree_node_tracker = unsorted_node_sample_tracker_->GetFeaturePartition(tree_num);
    data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
    data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);
    data_size_t idx;
    auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
    auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;

    // Initialize all sufficient statistics
    ForestLeafSuffStatType root_suff_stat = ForestLeafSuffStatType();
    ForestLeafSuffStatType left_suff_stat = ForestLeafSuffStatType();
    ForestLeafSuffStatType right_suff_stat = ForestLeafSuffStatType();
    
    // Iterate through every observation in the node
    double feature_value;
    for (auto i = node_begin_iter; i != node_end_iter; i++) {
      idx = *i;
      feature_value = data->covariates(idx, feature_split);
      root_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, idx);
      if (SplitTrueNumeric(feature_value, split_value)) {
        // Increment new left node sufficient statistic if split is true
        left_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, idx);
      } else {
        // Increment new left node sufficient statistic if split is false
        right_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, idx);
      }
    }

    double split_log_ml = SplitLogMarginalLikelihood<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(left_suff_stat, right_suff_stat, *leaf_prior, outcome_variance);
    double no_split_log_ml = NoSplitLogMarginalLikelihood<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(root_suff_stat, *leaf_prior, outcome_variance);

    return std::tuple<double, double, int, int>(split_log_ml, no_split_log_ml, left_suff_stat.n, right_suff_stat.n);
  }

  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  std::tuple<double, double, int32_t, int32_t> EvaluateSplitAndNoSplit(Tree* tree, UnivariateResidual* residual, ForestDatasetType* data, ForestLeafPriorType* leaf_prior, TreePrior* tree_prior, double outcome_variance, int32_t tree_num, int leaf_split, int feature_split, std::vector<std::uint32_t>& split_categories) {
    // Unpack shifted iterators to observations in a given node
    auto tree_node_tracker = unsorted_node_sample_tracker_->GetFeaturePartition(tree_num);
    data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
    data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);
    data_size_t idx;
    auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
    auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;

    // Initialize all sufficient statistics
    ForestLeafSuffStatType root_suff_stat = ForestLeafSuffStatType();
    ForestLeafSuffStatType left_suff_stat = ForestLeafSuffStatType();
    ForestLeafSuffStatType right_suff_stat = ForestLeafSuffStatType();
    
    // Iterate through every observation in the node
    double feature_value;
    for (auto i = node_begin_iter; i != node_end_iter; i++) {
      idx = *i;
      feature_value = data->covariates(idx, feature_split);

      // Increment sufficient statistics for the split node, regardless of covariate value
      root_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, idx);
      if (SplitTrueCategorical(feature_value, split_categories)) {
        // Increment new left node sufficient statistic if split is true
        left_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, idx);
      } else {
        // Increment new left node sufficient statistic if split is false
        right_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, idx);
      }
    }

    double split_log_ml = SplitLogMarginalLikelihood<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(left_suff_stat, right_suff_stat, *leaf_prior, outcome_variance);
    double no_split_log_ml = NoSplitLogMarginalLikelihood<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(root_suff_stat, *leaf_prior, outcome_variance);

    return std::tuple<double, double, int, int>(split_log_ml, no_split_log_ml, left_suff_stat.n, right_suff_stat.n);
  }

  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  std::tuple<double, double, int32_t, int32_t> EvaluateSplitAndNoSplit(Tree* tree, UnivariateResidual* residual, ForestDatasetType* data, ForestLeafPriorType* leaf_prior, TreePrior* tree_prior, double outcome_variance, int32_t tree_num, int parent_node, int left_node, int right_node) {
    // Unpack shifted iterators to observations in a given node
    auto tree_node_tracker = unsorted_node_sample_tracker_->GetFeaturePartition(tree_num);
    data_size_t left_node_begin = tree_node_tracker->NodeBegin(left_node);
    data_size_t left_node_end = tree_node_tracker->NodeEnd(left_node);
    data_size_t right_node_begin = tree_node_tracker->NodeBegin(right_node);
    data_size_t right_node_end = tree_node_tracker->NodeEnd(right_node);
    data_size_t idx;
    auto left_node_begin_iter = tree_node_tracker->indices_.begin() + left_node_begin;
    auto left_node_end_iter = tree_node_tracker->indices_.begin() + left_node_end;
    auto right_node_begin_iter = tree_node_tracker->indices_.begin() + right_node_begin;
    auto right_node_end_iter = tree_node_tracker->indices_.begin() + right_node_end;

    // Initialize all sufficient statistics
    ForestLeafSuffStatType root_suff_stat = ForestLeafSuffStatType();
    ForestLeafSuffStatType left_suff_stat = ForestLeafSuffStatType();
    ForestLeafSuffStatType right_suff_stat = ForestLeafSuffStatType();
    
    double feature_value;
    // Update left node sufficient statistics
    for (auto i = left_node_begin_iter; i != left_node_end_iter; i++) {
      idx = *i;
      // Increment sufficient statistics for the parent and left nodes
      root_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, idx);
      left_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, idx);
    }
    // Update right node sufficient statistics
    for (auto i = right_node_begin_iter; i != right_node_end_iter; i++) {
      idx = *i;
      // Increment sufficient statistics for the parent and left nodes
      root_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, idx);
      right_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, idx);
    }

    double split_log_ml = SplitLogMarginalLikelihood<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(left_suff_stat, right_suff_stat, *leaf_prior, outcome_variance);
    double no_split_log_ml = NoSplitLogMarginalLikelihood<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(root_suff_stat, *leaf_prior, outcome_variance);

    return std::tuple<double, double, int, int>(split_log_ml, no_split_log_ml, left_suff_stat.n, right_suff_stat.n);
  }

  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  bool NodesNonConstantAfterSplit(ForestDatasetType* data, Tree* tree, int leaf_split, int feature_split, double split_value, int tree_num) {
    int p = data->covariates.cols();
    data_size_t idx;
    double feature_value;
    double split_feature_value;
    double var_max_left;
    double var_min_left;
    double var_max_right;
    double var_min_right;
    auto tree_node_tracker = unsorted_node_sample_tracker_->GetFeaturePartition(tree_num);
    data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
    data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);

    for (int j = 0; j < p; j++) {
      var_max_left = std::numeric_limits<double>::min();
      var_min_left = std::numeric_limits<double>::max();
      var_max_right = std::numeric_limits<double>::min();
      var_min_right = std::numeric_limits<double>::max();
      auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
      auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
      for (auto i = node_begin_iter; i != node_end_iter; i++) {
        idx = *i;
        feature_value = data->covariates(idx, j);
        split_feature_value = data->covariates(idx, feature_split);
        if (SplitTrueNumeric(split_feature_value, split_value)) {
          if (var_max_left < feature_value) {
            var_max_left = feature_value;
          } else if (var_min_left > feature_value) {
            var_min_left = feature_value;
          }
        } else {
          if (var_max_right < feature_value) {
            var_max_right = feature_value;
          } else if (var_min_right > feature_value) {
            var_min_right = feature_value;
          }
        }
      }
      if ((var_max_left > var_min_left) && (var_max_right > var_min_right)) {
        return true;
      }
    }
    return false;
  }
  
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  bool NodesNonConstantAfterSplit(ForestDatasetType* data, Tree* tree, int leaf_split, int feature_split, std::vector<std::uint32_t> split_categories, int tree_num) {
    int p = data->covariates.cols();
    data_size_t idx;
    double feature_value;
    double split_feature_value;
    double var_max_left;
    double var_min_left;
    double var_max_right;
    double var_min_right;
    auto tree_node_tracker = unsorted_node_sample_tracker_->GetFeaturePartition(tree_num);
    data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
    data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);

    for (int j = 0; j < p; j++) {
      var_max_left = std::numeric_limits<double>::min();
      var_min_left = std::numeric_limits<double>::max();
      var_max_right = std::numeric_limits<double>::min();
      var_min_right = std::numeric_limits<double>::max();
      auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
      auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
      for (auto i = node_begin_iter; i != node_end_iter; i++) {
        idx = *i;
        feature_value = data->covariates(idx, j);
        split_feature_value = data->covariates(idx, feature_split);
        if (SplitTrueCategorical(split_feature_value, split_categories)) {
          if (var_max_left < feature_value) {
            var_max_left = feature_value;
          } else if (var_min_left > feature_value) {
            var_min_left = feature_value;
          }
        } else {
          if (var_max_right < feature_value) {
            var_max_right = feature_value;
          } else if (var_min_right > feature_value) {
            var_min_right = feature_value;
          }
        }
      }
      if ((var_max_left > var_min_left) && (var_max_right > var_min_right)) {
        return true;
      }
    }
    return false;
  }
};

class GFRTreeSampler : public TreeSampler {
 public:
  GFRTreeSampler() {}
  ~GFRTreeSampler() {}
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  void Initialize(ForestDatasetType* data, int num_trees, int num_observations, std::vector<FeatureType>& feature_types) {
    sample_node_mapper_ = std::make_unique<SampleNodeMapper>(num_trees, num_observations);
    presort_container_ = std::make_unique<FeaturePresortRootContainer>(data->covariates, feature_types);
    sorted_node_sample_tracker_ = std::make_unique<SortedNodeSampleTracker>(presort_container_.get(), data->covariates, feature_types);
    for (int i = 0; i < num_trees; i++) {
      AssignAllSamplesToRoot(i);
    }
  }
  
  /*! \brief Perform one stochastic grow-from-root step for a tree */
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  void SampleTree(Tree* tree, UnivariateResidual* residual, ForestDatasetType* data, ForestLeafPriorType* leaf_prior, TreePrior* tree_prior, double outcome_variance, std::mt19937& gen, int32_t tree_num, int32_t cutpoint_grid_size, std::vector<FeatureType>& feature_types) {
    int root_id = Tree::kRoot;
    int curr_node_id;
    data_size_t curr_node_begin;
    data_size_t curr_node_end;
    data_size_t n = data->covariates.rows();
    // Mapping from node id to start and end points of sorted indices
    std::unordered_map<int, std::pair<data_size_t, data_size_t>> node_index_map_;
    node_index_map_.insert({root_id, std::make_pair(0, n)});
    std::pair<data_size_t, data_size_t> begin_end;
    // Add root node to the split queue
    std::deque<node_t> split_queue_;
    split_queue_.push_back(Tree::kRoot);
    // Run the "GrowFromRoot" procedure using a stack in place of recursion
    while (!split_queue_.empty()) {
      // Remove the next node from the queue
      curr_node_id = split_queue_.front();
      split_queue_.pop_front();
      // Determine the beginning and ending indices of the left and right nodes
      begin_end = node_index_map_[curr_node_id];
      curr_node_begin = begin_end.first;
      curr_node_end = begin_end.second;
      // Draw a split rule at random
      SampleSplitRule<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(tree, residual, data, leaf_prior, tree_prior, sorted_node_sample_tracker_.get(), sample_node_mapper_.get(), feature_types, gen, outcome_variance, tree_num, curr_node_id, cutpoint_grid_size, curr_node_begin, curr_node_end, split_queue_, node_index_map_);
    }
  }
  
  void AssignAllSamplesToRoot(int tree_num);
  data_size_t GetNodeId(int observation_num, int tree_num);
  
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  void Reset(TreeEnsembleContainer* container, ForestDatasetType* data, std::vector<FeatureType>& feature_types, int tree_num, int sample_num, int prev_sample_num) {
    // Reset training data so that features are pre-sorted based on the entire dataset
    sorted_node_sample_tracker_.reset(new SortedNodeSampleTracker(presort_container_.get(), data->covariates, feature_types));
    // Reset tree j to a constant root node
    (container->GetEnsemble(sample_num))->ResetInitTree(tree_num);
    // Reset the observation indices to point to node 0
    AssignAllSamplesToRoot(tree_num);
  }
  
  SortedNodeSampleTracker* GetNodeSampleTracker() {return sorted_node_sample_tracker_.get();}
 private:
  /*! \brief Mapper from observations to leaf node indices for every tree in a forest */
  std::unique_ptr<SampleNodeMapper> sample_node_mapper_;
  /*! \brief Data structure tracking / updating observations available in each node for every tree in a forest */
  std::unique_ptr<FeaturePresortRootContainer> presort_container_;
  std::unique_ptr<SortedNodeSampleTracker> sorted_node_sample_tracker_;

  /*! \brief Sample a split (or no-split) rule at a given tree node during the recursive grow-from-root algorithm */
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  void SampleSplitRule(Tree* tree, UnivariateResidual* residual, ForestDatasetType* data, ForestLeafPriorType* leaf_prior, TreePrior* tree_prior, SortedNodeSampleTracker* sorted_node_sample_tracker, SampleNodeMapper* sample_node_mapper, std::vector<FeatureType>& feature_types, std::mt19937& gen, double outcome_variance, int32_t tree_num, int leaf_node, int32_t cutpoint_grid_size, data_size_t node_begin, data_size_t node_end, std::deque<node_t>& split_queue, std::unordered_map<int, std::pair<data_size_t, data_size_t>>& node_index_map) {
    std::vector<double> log_cutpoint_evaluations;
    std::vector<int> cutpoint_features;
    std::vector<double> cutpoint_values;
    std::vector<FeatureType> cutpoint_feature_types;
    StochTree::data_size_t valid_cutpoint_count;
    CutpointGridContainer cutpoint_grid_container(data->covariates, residual->residual, cutpoint_grid_size);
    Cutpoints<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(
      tree, residual, data, leaf_prior, tree_prior, sorted_node_sample_tracker, sample_node_mapper, feature_types, 
      gen, outcome_variance, tree_num, leaf_node, cutpoint_grid_size, node_begin, node_end, split_queue, node_index_map, 
      log_cutpoint_evaluations, cutpoint_features, cutpoint_values, cutpoint_feature_types, valid_cutpoint_count, cutpoint_grid_container);
    
    // Convert log marginal likelihood to marginal likelihood, normalizing by the maximum log-likelihood
    double largest_mll = *std::max_element(log_cutpoint_evaluations.begin(), log_cutpoint_evaluations.end());
    std::vector<double> cutpoint_evaluations(log_cutpoint_evaluations.size());
    for (data_size_t i = 0; i < log_cutpoint_evaluations.size(); i++){
      cutpoint_evaluations[i] = std::exp(log_cutpoint_evaluations[i] - largest_mll);
    }
    
    // Sample the split (including a "no split" option)
    std::discrete_distribution<data_size_t> split_dist(cutpoint_evaluations.begin(), cutpoint_evaluations.end());
    data_size_t split_chosen = split_dist(gen);
    if (split_chosen == valid_cutpoint_count){
      // "No split" sampled, don't split or add any nodes to split queue
      return;
    } else {
      // Split sampled
      int feature_split = cutpoint_features[split_chosen];
      FeatureType feature_type = cutpoint_feature_types[split_chosen];
      double split_value = cutpoint_values[split_chosen];
      // Perform all of the relevant "split" operations in the model, tree and training dataset
      AddSplitToModel<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(
        tree, residual, data, leaf_prior, tree_prior, 
        sorted_node_sample_tracker, sample_node_mapper, feature_type, 
        leaf_node, node_begin, node_end, feature_split, split_value, split_queue, 
        tree_num, cutpoint_grid_container, node_index_map
      );
    }
  }

  /*! \brief Evaluate potential splits for each variable */
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  void Cutpoints(Tree* tree, UnivariateResidual* residual, ForestDatasetType* data, ForestLeafPriorType* leaf_prior, TreePrior* tree_prior, 
                 SortedNodeSampleTracker* sorted_node_sample_tracker, SampleNodeMapper* sample_node_mapper, std::vector<FeatureType>& feature_types, 
                 std::mt19937& gen, double outcome_variance, int32_t tree_num, int leaf_node, int32_t cutpoint_grid_size, data_size_t node_begin, 
                 data_size_t node_end, std::deque<node_t>& split_queue, std::unordered_map<int, std::pair<data_size_t, data_size_t>>& node_index_map, 
                 std::vector<double>& log_cutpoint_evaluations, std::vector<int>& cutpoint_features, std::vector<double>& cutpoint_values, 
                 std::vector<FeatureType>& cutpoint_feature_types, data_size_t& valid_cutpoint_count, CutpointGridContainer& cutpoint_grid_container) {
    // Compute sufficient statistics for the current node
    ForestLeafSuffStatType root_suff_stat = ForestLeafSuffStatType();
    data_size_t sort_idx;
    for (data_size_t i = node_begin; i < node_end; i++) {
      sort_idx = sorted_node_sample_tracker->SortIndex(i, 0);
      root_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, sort_idx);
    }
    
    // Evaluate the "no split" integrated likelihood
    double no_split_log_ml = NoSplitLogMarginalLikelihood<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(root_suff_stat, *leaf_prior, outcome_variance);
    
    // Declare the split_log_ml variable and initialize left and right suff stats (to default values)
    double split_log_ml;
    ForestLeafSuffStatType left_suff_stat = ForestLeafSuffStatType();
    ForestLeafSuffStatType right_suff_stat = ForestLeafSuffStatType();

    // Clear vectors
    log_cutpoint_evaluations.clear();
    cutpoint_features.clear();
    cutpoint_values.clear();
    cutpoint_feature_types.clear();

    // Reset cutpoint grid container
    cutpoint_grid_container.Reset(data->covariates, residual->residual, cutpoint_grid_size);

    // Compute sufficient statistics for each possible split
    data_size_t num_cutpoints = 0;
    bool valid_split = false;
    data_size_t node_row_iter;
    data_size_t current_bin_begin, current_bin_size, next_bin_begin;
    data_size_t feature_sort_idx;
    data_size_t row_iter_idx;
    double outcome_val, outcome_val_sq;
    FeatureType feature_type;
    double feature_value = 0.0;
    double cutoff_value = 0.0;
    double log_split_eval = 0.0;
    for (int j = 0; j < data->covariates.cols(); j++) {

      // Enumerate cutpoint strides
      cutpoint_grid_container.CalculateStrides(data->covariates, residual->residual, sorted_node_sample_tracker, leaf_node, node_begin, node_end, j, feature_types);

      // Iterate through possible cutpoints
      int32_t num_feature_cutpoints = cutpoint_grid_container.NumCutpoints(j);
      feature_type = feature_types[j];
      for (data_size_t cutpoint_idx = 0; cutpoint_idx < (num_feature_cutpoints - 1); cutpoint_idx++) {
        // Unpack cutpoint details, noting that since we partition an entire cutpoint bin to the left, 
        // we must stop one bin before the total number of cutpoint bins
        current_bin_begin = cutpoint_grid_container.BinStartIndex(cutpoint_idx, j);
        current_bin_size = cutpoint_grid_container.BinLength(cutpoint_idx, j);
        next_bin_begin = cutpoint_grid_container.BinStartIndex(cutpoint_idx + 1, j);

        // Accumulate sufficient statistics for the left node
        for (data_size_t k = 0; k < current_bin_size; k++) {
          row_iter_idx = current_bin_begin + k;
          feature_sort_idx = sorted_node_sample_tracker->SortIndex(row_iter_idx, j);
          left_suff_stat.template IncrementSuffStat<ForestDatasetType>(data, residual, feature_sort_idx);
        }

        // Compute the corresponding right node sufficient statistics
        right_suff_stat.SubtractSuffStat(root_suff_stat, left_suff_stat);

        // Store the bin index as the "cutpoint value" - we can use this to query the actual split 
        // value or the set of split categories later on once a split is chose
        cutoff_value = cutpoint_idx;

        // Only include cutpoint for consideration if it defines a valid split in the training data
        int32_t min_samples_in_leaf = tree_prior->GetMinSamplesLeaf();
        valid_split = (left_suff_stat.SampleGreaterThan(min_samples_in_leaf) && 
                       right_suff_stat.SampleGreaterThan(min_samples_in_leaf));
        if (valid_split) {
          num_cutpoints++;
          // Add to split rule vector
          cutpoint_feature_types.push_back(feature_type);
          cutpoint_features.push_back(j);
          cutpoint_values.push_back(cutoff_value);
          // Add the log marginal likelihood of the split to the split eval vector 
          split_log_ml = SplitLogMarginalLikelihood<ForestDatasetType, ForestLeafPriorType, ForestLeafSuffStatType>(left_suff_stat, right_suff_stat, *leaf_prior, outcome_variance);
          log_cutpoint_evaluations.push_back(split_log_ml);
        }
      }
    }

    // Add the log marginal likelihood of the "no-split" option (adjusted for tree prior and cutpoint size per the XBART paper)
    cutpoint_features.push_back(-1);
    cutpoint_values.push_back(std::numeric_limits<double>::max());
    cutpoint_feature_types.push_back(FeatureType::kNumeric);
    
    // Compute an adjustment to reflect the no split prior probability and the number of cutpoints
    double bart_prior_no_split_adj;
    double alpha = tree_prior->GetAlpha();
    double beta = tree_prior->GetBeta();
    int node_depth = tree->GetDepth(leaf_node);
    if (num_cutpoints == 0) {
      bart_prior_no_split_adj = std::log(((std::pow(1+node_depth, beta))/alpha) - 1.0);
    } else {
      bart_prior_no_split_adj = std::log(((std::pow(1+node_depth, beta))/alpha) - 1.0) + std::log(num_cutpoints);
    }
    no_split_log_ml += bart_prior_no_split_adj;
    
    // Add the no split evaluation to the evaluations
    log_cutpoint_evaluations.push_back(no_split_log_ml);
    valid_cutpoint_count = num_cutpoints;
  }

  /*! \brief Add a split to a model in the GFR algorithm */
  template <typename ForestDatasetType, typename ForestLeafPriorType, typename ForestLeafSuffStatType>
  void AddSplitToModel(Tree* tree, UnivariateResidual* residual, ForestDatasetType* data, ForestLeafPriorType* leaf_prior, TreePrior* tree_prior, 
                       SortedNodeSampleTracker* sorted_node_sample_tracker, SampleNodeMapper* sample_node_mapper, FeatureType feature_type, 
                       node_t leaf_node, data_size_t node_begin, data_size_t node_end, int feature_split, double split_value, std::deque<node_t>& split_queue, 
                       int tree_num, CutpointGridContainer& cutpoint_grid_container, std::unordered_map<int, std::pair<data_size_t, data_size_t>>& node_index_map) {
    // Compute node sample size
    data_size_t node_n = node_end - node_begin;
    
    // Actual numeric cutpoint used for ordered categorical and numeric features
    double split_value_numeric;
    
    // We will use these later in the model expansion
    data_size_t left_n = 0;
    data_size_t sort_idx;
    double feature_value;
    bool split_true;

    // Split the tree at leaf node
    // Use 0 as a "temporary" leaf value since we sample 
    // all leaf parameters after tree sampling is complete
    bool tree_multivariate = false;
    if (std::is_same_v<RegressionLeafForestDataset, ForestDatasetType>) {
      RegressionLeafForestDataset* basis_data_ptr = dynamic_cast<RegressionLeafForestDataset*>(data);
      if (basis_data_ptr->covariates.cols() > 1) {
        tree_multivariate = true;
      }
    }
    double left_leaf_value = 0.;
    double right_leaf_value = 0.;
    std::vector<double> left_leaf_vector, right_leaf_vector;
    if (tree_multivariate) {
      int basis_dim = dynamic_cast<RegressionLeafForestDataset*>(data)->covariates.cols();
      left_leaf_vector.resize(basis_dim, 0.);
      right_leaf_vector.resize(basis_dim, 0.);
    }

    if (feature_type == FeatureType::kUnorderedCategorical) {
      // Determine the number of categories available in a categorical split and the set of categories that route observations to the left node after split
      int num_categories;
      std::vector<std::uint32_t> categories = cutpoint_grid_container.CutpointVector(static_cast<std::uint32_t>(split_value), feature_split);

      // Determine the number of observation in the newly created left node
      for (data_size_t i = node_begin; i < node_end; i++) {
        sort_idx = sorted_node_sample_tracker->SortIndex(i, feature_split);
        feature_value = data->covariates(sort_idx, feature_split);
        split_true = SplitTrueCategorical(feature_value, categories);
        if (split_true) left_n += 1;
      }

      // Perform the split
      if (tree_multivariate) {
        tree->ExpandNode(leaf_node, feature_split, categories, true, left_leaf_value, right_leaf_value);
      } else {
        tree->ExpandNode(leaf_node, feature_split, categories, true, left_leaf_vector, right_leaf_vector);
      }
      // Partition the dataset according to the new split rule and determine the beginning and end of the new left and right nodes
      sorted_node_sample_tracker->PartitionNode(data->covariates, leaf_node, feature_split, categories);
    } else if (feature_type == FeatureType::kOrderedCategorical) {
      // Convert the bin split to an actual split value
      // split_value_numeric = cutpoint_grid_container->CutpointValue(static_cast<std::uint32_t>(split_value), feature_split);
      split_value_numeric = cutpoint_grid_container.CutpointValue(static_cast<std::uint32_t>(split_value), feature_split);

      // Determine the number of observation in the newly created left node
      for (data_size_t i = node_begin; i < node_end; i++) {
        sort_idx = sorted_node_sample_tracker->SortIndex(i, feature_split);
        feature_value = data->covariates(sort_idx, feature_split);
        split_true = SplitTrueNumeric(feature_value, split_value_numeric);
        if (split_true) left_n += 1;
      }
      
      if (tree_multivariate) {
        tree->ExpandNode(leaf_node, feature_split, split_value_numeric, true, left_leaf_value, right_leaf_value);
      } else {
        tree->ExpandNode(leaf_node, feature_split, split_value_numeric, true, left_leaf_vector, right_leaf_vector);
      }
      // Partition the dataset according to the new split rule and determine the beginning and end of the new left and right nodes
      sorted_node_sample_tracker->PartitionNode(data->covariates, leaf_node, feature_split, split_value_numeric);
    } else if (feature_type == FeatureType::kNumeric) {
      // Convert the bin split to an actual split value
      split_value_numeric = cutpoint_grid_container.CutpointValue(static_cast<std::uint32_t>(split_value), feature_split);

      // Determine the number of observation in the newly created left node
      for (data_size_t i = node_begin; i < node_end; i++) {
        sort_idx = sorted_node_sample_tracker->SortIndex(i, feature_split);
        feature_value = data->covariates(sort_idx, feature_split);
        split_true = SplitTrueNumeric(feature_value, split_value_numeric);
        if (split_true) left_n += 1;
      }
      
      if (tree_multivariate) {
        tree->ExpandNode(leaf_node, feature_split, split_value_numeric, true, left_leaf_value, right_leaf_value);
      } else {
        tree->ExpandNode(leaf_node, feature_split, split_value_numeric, true, left_leaf_vector, right_leaf_vector);
      }
      // Partition the dataset according to the new split rule and determine the beginning and end of the new left and right nodes
      sorted_node_sample_tracker->PartitionNode(data->covariates, leaf_node, feature_split, split_value_numeric);
    } else {
      Log::Fatal("Invalid split type");
    }
    int left_node = tree->LeftChild(leaf_node);
    int right_node = tree->RightChild(leaf_node);

    // Update the leaf node observation tracker
    sorted_node_sample_tracker->UpdateObservationMapping(left_node, tree_num, sample_node_mapper);
    sorted_node_sample_tracker->UpdateObservationMapping(right_node, tree_num, sample_node_mapper);

    // Add the begin and end indices for the new left and right nodes to node_index_map
    node_index_map.insert({left_node, std::make_pair(node_begin, node_begin + left_n)});
    node_index_map.insert({right_node, std::make_pair(node_begin + left_n, node_end)});

    // Add the left and right nodes to the split tracker
    split_queue.push_front(right_node);
    split_queue.push_front(left_node);
  }
};

class LeafGaussianSampler {
 public:
  LeafGaussianSampler() {}
  virtual ~LeafGaussianSampler() = default;
  virtual void SampleLeafParameters(LeafConstantGaussianPrior* leaf_prior, ConstantLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {}
  virtual void SampleLeafParameters(LeafUnivariateRegressionGaussianPrior* leaf_prior, RegressionLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {}
  virtual void SampleLeafParameters(LeafMultivariateRegressionGaussianPrior* leaf_prior, RegressionLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {}
  virtual void SampleLeafParameters(LeafConstantGaussianPrior* leaf_prior, ConstantLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, SortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {}
  virtual void SampleLeafParameters(LeafUnivariateRegressionGaussianPrior* leaf_prior, RegressionLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, SortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {}
  virtual void SampleLeafParameters(LeafMultivariateRegressionGaussianPrior* leaf_prior, RegressionLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, SortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {}
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class LeafConstantGaussianSampler : public LeafGaussianSampler {
 public:
  LeafConstantGaussianSampler() {}
  ~LeafConstantGaussianSampler() {}
  double PosteriorMean(LeafConstantGaussianPrior* leaf_prior, LeafConstantGaussianSuffStat& leaf_suff_stat, double global_variance) {
    double tau = leaf_prior->GetPriorScale();
    double n = static_cast<double>(leaf_suff_stat.n);
    double sum_y = leaf_suff_stat.sum_y;
    return ((tau*sum_y)/(global_variance + (tau*n)));
  }
  double PosteriorVariance(LeafConstantGaussianPrior* leaf_prior, LeafConstantGaussianSuffStat& leaf_suff_stat, double global_variance) {
    double tau = leaf_prior->GetPriorScale();
    double n = static_cast<double>(leaf_suff_stat.n);
    return ((tau*global_variance)/(global_variance + (tau*n)));
  }
  void SampleLeafParameters(LeafConstantGaussianPrior* leaf_prior, ConstantLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {
    std::vector<int> tree_leaves = tree->GetLeaves();
    data_size_t node_begin, node_end;
    auto tree_node_tracker = node_sample_tracker->GetFeaturePartition(tree_num);
    data_size_t idx;
    LeafConstantGaussianSuffStat leaf_suff_stat;
    int leaf_num;
    double posterior_leaf_mean, posterior_leaf_variance;
    std::normal_distribution<double> leaf_node_dist(0.,1.);
    double leaf_value;

    for (int i = 0; i < tree_leaves.size(); i++) {
      // Compute node sufficient statistics
      leaf_num = tree_leaves[i];
      node_begin = tree_node_tracker->NodeBegin(leaf_num);
      node_end = tree_node_tracker->NodeEnd(leaf_num);
      auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
      auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
      leaf_suff_stat = LeafConstantGaussianSuffStat();
      
      for (auto i = node_begin_iter; i != node_end_iter; i++) {
        idx = *i;
        // Increment sufficient statistics for the split node, regardless of covariate value
        leaf_suff_stat.template IncrementSuffStat<ConstantLeafForestDataset>(data, residual, idx);
      }

      // Sample leaf value / vector and place it directly in the tree leaf
      posterior_leaf_mean = PosteriorMean(leaf_prior, leaf_suff_stat, global_variance);
      posterior_leaf_variance = PosteriorVariance(leaf_prior, leaf_suff_stat, global_variance);
      leaf_value = posterior_leaf_mean + std::sqrt(posterior_leaf_variance) * leaf_node_dist(gen);
      tree->SetLeaf(leaf_num, leaf_value);
    }
  }
  void SampleLeafParameters(LeafConstantGaussianPrior* leaf_prior, ConstantLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, SortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {
    std::vector<int> tree_leaves = tree->GetLeaves();
    data_size_t node_begin, node_end;
    LeafConstantGaussianSuffStat leaf_suff_stat;
    int leaf_num;
    double posterior_leaf_mean, posterior_leaf_variance;
    std::normal_distribution<double> leaf_node_dist(0.,1.);
    double leaf_value;

    for (int i = 0; i < tree_leaves.size(); i++) {
      // Compute node sufficient statistics
      leaf_num = tree_leaves[i];
      leaf_suff_stat = LeafConstantGaussianSuffStat();
      node_begin = node_sample_tracker->NodeBegin(leaf_num, 0);
      node_end = node_sample_tracker->NodeEnd(leaf_num, 0);
      data_size_t sort_idx;
      for (data_size_t i = node_begin; i < node_end; i++) {
        sort_idx = node_sample_tracker->SortIndex(i, 0);
        leaf_suff_stat.template IncrementSuffStat<ConstantLeafForestDataset>(data, residual, sort_idx);
      }

      // Sample leaf value / vector and place it directly in the tree leaf
      posterior_leaf_mean = PosteriorMean(leaf_prior, leaf_suff_stat, global_variance);
      posterior_leaf_variance = PosteriorVariance(leaf_prior, leaf_suff_stat, global_variance);
      leaf_value = posterior_leaf_mean + std::sqrt(posterior_leaf_variance) * leaf_node_dist(gen);
      tree->SetLeaf(leaf_num, leaf_value);
    }
  }
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class LeafUnivariateRegressionGaussianSampler : public LeafGaussianSampler {
 public:
  LeafUnivariateRegressionGaussianSampler() {}
  ~LeafUnivariateRegressionGaussianSampler() {}
  double PosteriorMean(LeafUnivariateRegressionGaussianPrior* leaf_prior, LeafUnivariateRegressionGaussianSuffStat& leaf_suff_stat, double global_variance) {
    double tau = leaf_prior->GetPriorScale();
    double sum_yx = leaf_suff_stat.sum_yx;
    double sum_x_squared = leaf_suff_stat.sum_x_squared;
    return ((tau*sum_yx)/(global_variance + (tau*sum_x_squared)));
  }
  double PosteriorVariance(LeafUnivariateRegressionGaussianPrior* leaf_prior, LeafUnivariateRegressionGaussianSuffStat& leaf_suff_stat, double global_variance) {
    double tau = leaf_prior->GetPriorScale();
    double sum_x_squared = leaf_suff_stat.sum_x_squared;
    return ((tau*global_variance)/(global_variance + (tau*sum_x_squared)));
  }
  void SampleLeafParameters(LeafUnivariateRegressionGaussianPrior* leaf_prior, RegressionLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {
    std::vector<int> tree_leaves = tree->GetLeaves();
    data_size_t node_begin, node_end;
    auto tree_node_tracker = node_sample_tracker->GetFeaturePartition(tree_num);
    data_size_t idx;
    LeafUnivariateRegressionGaussianSuffStat leaf_suff_stat;
    int leaf_num;
    double posterior_leaf_mean, posterior_leaf_variance;
    std::normal_distribution<double> leaf_node_dist(0.,1.);
    double leaf_value;

    for (int i = 0; i < tree_leaves.size(); i++) {
      // Compute node sufficient statistics
      leaf_num = tree_leaves[i];
      node_begin = tree_node_tracker->NodeBegin(leaf_num);
      node_end = tree_node_tracker->NodeEnd(leaf_num);
      auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
      auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
      leaf_suff_stat = LeafUnivariateRegressionGaussianSuffStat();
      
      for (auto i = node_begin_iter; i != node_end_iter; i++) {
        idx = *i;
        // Increment sufficient statistics for the split node, regardless of covariate value
        leaf_suff_stat.template IncrementSuffStat<RegressionLeafForestDataset>(data, residual, idx);
      }

      // Sample leaf value / vector and place it directly in the tree leaf
      posterior_leaf_mean = PosteriorMean(leaf_prior, leaf_suff_stat, global_variance);
      posterior_leaf_variance = PosteriorVariance(leaf_prior, leaf_suff_stat, global_variance);
      leaf_value = posterior_leaf_mean + std::sqrt(posterior_leaf_variance) * leaf_node_dist(gen);
      tree->SetLeaf(leaf_num, leaf_value);
    }
  }
  void SampleLeafParameters(LeafUnivariateRegressionGaussianPrior* leaf_prior, RegressionLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, SortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {
    std::vector<int> tree_leaves = tree->GetLeaves();
    data_size_t node_begin, node_end;
    LeafUnivariateRegressionGaussianSuffStat leaf_suff_stat;
    int leaf_num;
    double posterior_leaf_mean, posterior_leaf_variance;
    std::normal_distribution<double> leaf_node_dist(0.,1.);
    double leaf_value;

    for (int i = 0; i < tree_leaves.size(); i++) {
      // Compute node sufficient statistics
      leaf_num = tree_leaves[i];
      leaf_suff_stat = LeafUnivariateRegressionGaussianSuffStat();
      node_begin = node_sample_tracker->NodeBegin(leaf_num, 0);
      node_end = node_sample_tracker->NodeEnd(leaf_num, 0);
      data_size_t sort_idx;
      for (data_size_t i = node_begin; i < node_end; i++) {
        sort_idx = node_sample_tracker->SortIndex(i, 0);
        leaf_suff_stat.template IncrementSuffStat<RegressionLeafForestDataset>(data, residual, sort_idx);
      }

      // Sample leaf value / vector and place it directly in the tree leaf
      posterior_leaf_mean = PosteriorMean(leaf_prior, leaf_suff_stat, global_variance);
      posterior_leaf_variance = PosteriorVariance(leaf_prior, leaf_suff_stat, global_variance);
      leaf_value = posterior_leaf_mean + std::sqrt(posterior_leaf_variance) * leaf_node_dist(gen);
      tree->SetLeaf(leaf_num, leaf_value);
    }
  }
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class LeafMultivariateRegressionGaussianSampler : public LeafGaussianSampler {
 public:
  LeafMultivariateRegressionGaussianSampler() {}
  ~LeafMultivariateRegressionGaussianSampler() {}
  Eigen::MatrixXd PosteriorMean(LeafMultivariateRegressionGaussianPrior* leaf_prior, LeafMultivariateRegressionGaussianSuffStat& leaf_suff_stat, double global_variance) {
    Eigen::MatrixXd Sigma = leaf_prior->GetPriorScale();
    Eigen::MatrixXd inverse_posterior_var = (Sigma.inverse().array() + (leaf_suff_stat.XtX/global_variance).array()).inverse();
    Eigen::MatrixXd result = inverse_posterior_var * (leaf_suff_stat.Xty / global_variance);
    return result;
  }
  Eigen::MatrixXd PosteriorVariance(LeafMultivariateRegressionGaussianPrior* leaf_prior, LeafMultivariateRegressionGaussianSuffStat& leaf_suff_stat, double global_variance) {
    Eigen::MatrixXd Sigma = leaf_prior->GetPriorScale();
    Eigen::MatrixXd result = Sigma.inverse().array() + (leaf_suff_stat.XtX/global_variance).array();
    return result;
  }
  void SampleLeafParameters(LeafMultivariateRegressionGaussianPrior* leaf_prior, RegressionLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, UnsortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {
    int basis_dim = data->basis.cols();
    std::vector<int> tree_leaves = tree->GetLeaves();
    data_size_t node_begin, node_end;
    auto tree_node_tracker = node_sample_tracker->GetFeaturePartition(tree_num);
    data_size_t idx;
    LeafMultivariateRegressionGaussianSuffStat leaf_suff_stat(basis_dim);
    int leaf_num;
    Eigen::MatrixXd mu_post, Sigma_post;
    std::normal_distribution<double> leaf_node_dist(0.,1.);
    Eigen::MatrixXd std_norm_vec(basis_dim, 1);

    for (int i = 0; i < tree_leaves.size(); i++) {
      // Compute node sufficient statistics
      leaf_num = tree_leaves[i];
      node_begin = tree_node_tracker->NodeBegin(leaf_num);
      node_end = tree_node_tracker->NodeEnd(leaf_num);
      auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
      auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
      leaf_suff_stat = LeafMultivariateRegressionGaussianSuffStat(basis_dim);
      
      for (auto i = node_begin_iter; i != node_end_iter; i++) {
        idx = *i;
        // Increment sufficient statistics for the split node, regardless of covariate value
        leaf_suff_stat.template IncrementSuffStat<RegressionLeafForestDataset>(data, residual, idx);
      }

      // Mean, variance, and variance cholesky decomposition
      mu_post = PosteriorMean(leaf_prior, leaf_suff_stat, global_variance);
      Sigma_post = PosteriorVariance(leaf_prior, leaf_suff_stat, global_variance);
      Eigen::LLT<Eigen::MatrixXd> decomposition(Sigma_post);
      Eigen::MatrixXd Sigma_post_chol = decomposition.matrixL();

      // Sample a vector of standard normal random variables
      for (int i = 0; i < basis_dim; i++) {
        std_norm_vec(i,0) = leaf_node_dist(gen);
      }

      // Generate the leaf parameters
      Eigen::MatrixXd leaf_values_raw = mu_post + Sigma_post_chol * std_norm_vec;
      std::vector<double> result(basis_dim);
      for (int i = 0; i < basis_dim; i++) {
        result[i] = leaf_values_raw(i, 0);
      }
      tree->SetLeafVector(leaf_num, result);
    }
  }
  void SampleLeafParameters(LeafMultivariateRegressionGaussianPrior* leaf_prior, RegressionLeafForestDataset* data, UnivariateResidual* residual, Tree* tree, SortedNodeSampleTracker* node_sample_tracker, int tree_num, std::mt19937& gen, double global_variance) {
    int basis_dim = data->basis.cols();
    std::vector<int> tree_leaves = tree->GetLeaves();
    data_size_t node_begin, node_end;
    LeafMultivariateRegressionGaussianSuffStat leaf_suff_stat(basis_dim);
    int leaf_num;
    Eigen::MatrixXd mu_post, Sigma_post;
    std::normal_distribution<double> leaf_node_dist(0.,1.);
    Eigen::MatrixXd std_norm_vec(basis_dim, 1);

    for (int i = 0; i < tree_leaves.size(); i++) {
      // Compute node sufficient statistics
      leaf_num = tree_leaves[i];
      leaf_suff_stat = LeafMultivariateRegressionGaussianSuffStat(basis_dim);
      node_begin = node_sample_tracker->NodeBegin(leaf_num, 0);
      node_end = node_sample_tracker->NodeEnd(leaf_num, 0);
      data_size_t sort_idx;
      for (data_size_t i = node_begin; i < node_end; i++) {
        sort_idx = node_sample_tracker->SortIndex(i, 0);
        leaf_suff_stat.template IncrementSuffStat<RegressionLeafForestDataset>(data, residual, sort_idx);
      }

      // Mean, variance, and variance cholesky decomposition
      mu_post = PosteriorMean(leaf_prior, leaf_suff_stat, global_variance);
      Sigma_post = PosteriorVariance(leaf_prior, leaf_suff_stat, global_variance);
      Eigen::LLT<Eigen::MatrixXd> decomposition(Sigma_post);
      Eigen::MatrixXd Sigma_post_chol = decomposition.matrixL();

      // Sample a vector of standard normal random variables
      for (int i = 0; i < basis_dim; i++) {
        std_norm_vec(i,0) = leaf_node_dist(gen);
      }

      // Generate the leaf parameters
      Eigen::MatrixXd leaf_values_raw = mu_post + Sigma_post_chol * std_norm_vec;
      std::vector<double> result(basis_dim);
      for (int i = 0; i < basis_dim; i++) {
        result[i] = leaf_values_raw(i, 0);
      }
      tree->SetLeafVector(leaf_num, result);
    }
  }
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class GlobalHomoskedasticVarianceSampler {
 public:
  GlobalHomoskedasticVarianceSampler() {}
  ~GlobalHomoskedasticVarianceSampler() {}
  double PosteriorShape(UnivariateResidual* residual, IGVariancePrior* variance_prior) {
    data_size_t n = residual->residual.rows();
    double a = variance_prior->GetShape();
    return (a/2.0) + n;
  }
  double PosteriorScale(UnivariateResidual* residual, IGVariancePrior* variance_prior) {
    double a = variance_prior->GetShape();
    double b = variance_prior->GetScale();
    data_size_t n = residual->residual.rows();
    double sum_sq_resid = 0.;
    for (data_size_t i = 0; i < n; i++) {
      sum_sq_resid += std::pow(residual->residual(i, 0), 2);
    }
    return (a*b/2.0) + sum_sq_resid;
  }
  double SampleVarianceParameter(UnivariateResidual* residual, IGVariancePrior* variance_prior, std::mt19937& gen) {
    double ig_shape = PosteriorShape(residual, variance_prior);
    double ig_scape = PosteriorScale(residual, variance_prior);

    // C++ standard library provides a gamma distribution with scale
    // parameter, but the correspondence between gamma and IG is that 
    // 1 / gamma(a,b) ~ IG(a,b) when b is a __rate__ parameter.
    // Before sampling, we convert ig_scale to a gamma scale parameter by 
    // taking its multiplicative inverse.
    double gamma_scale = 1./ig_scape;
    std::gamma_distribution<double> residual_variance_dist(ig_shape, gamma_scale);
    return (1/residual_variance_dist(gen));
  }
};

/*! \brief Marginal likelihood and posterior computation for gaussian homoskedastic constant leaf outcome model */
class LeafNodeHomoskedasticVarianceSampler {
 public:
  LeafNodeHomoskedasticVarianceSampler() {}
  ~LeafNodeHomoskedasticVarianceSampler() {}
  double PosteriorShape(TreeEnsemble* forest, IGVariancePrior* variance_prior) {
    double a = variance_prior->GetShape();
    data_size_t num_leaves = forest->NumLeaves();
    return (a/2.0) + num_leaves;
  }
  double PosteriorScale(TreeEnsemble* forest, IGVariancePrior* variance_prior) {
    double a = variance_prior->GetShape();
    double b = variance_prior->GetScale();
    double mu_sq = forest->SumLeafSquared();
    return (b/2.0) + mu_sq;
  }
  double SampleVarianceParameter(TreeEnsemble* forest, IGVariancePrior* variance_prior, std::mt19937& gen) {
    double ig_shape = PosteriorShape(forest, variance_prior);
    double ig_scape = PosteriorScale(forest, variance_prior);

    // C++ standard library provides a gamma distribution with scale
    // parameter, but the correspondence between gamma and IG is that 
    // 1 / gamma(a,b) ~ IG(a,b) when b is a __rate__ parameter.
    // Before sampling, we convert ig_scale to a gamma scale parameter by 
    // taking its multiplicative inverse.
    double gamma_scale = 1./ig_scape;
    std::gamma_distribution<double> residual_variance_dist(ig_shape, gamma_scale);
    return (1/residual_variance_dist(gen));
  }
};

/*! \brief Forward declaration of RandomEffectsPersisted class */
class RandomEffectsPersisted;

/*! \brief Sampling, prediction, and group / component tracking for random effects */
class RandomEffectsSampler {
 friend RandomEffectsPersisted;
 public:
  RandomEffectsSampler() {}
  // RandomEffectsSampler(std::vector<int32_t>& group_labels, int32_t num_components, int32_t num_groups);
  RandomEffectsSampler(RegressionRandomEffectsDataset* rfx_dataset, RandomEffectsRegressionGaussianPrior* rfx_prior);
  void InitializeParameters(RegressionRandomEffectsDataset* rfx_dataset, UnivariateResidual* residual);
  void SampleRandomEffects(RandomEffectsRegressionGaussianPrior* rfx_prior, RegressionRandomEffectsDataset* rfx_dataset, UnivariateResidual* residual, std::mt19937& gen);
  ~RandomEffectsSampler() {}
  std::vector<std::int32_t> GroupObservationIndices(std::int32_t group_num) const;
  void InitializeParameters(Eigen::MatrixXd& X, Eigen::MatrixXd& y);
  void SampleRandomEffects(Eigen::MatrixXd& X, Eigen::VectorXd& y, std::mt19937& gen, double a, double b);
  void SampleAlpha(Eigen::MatrixXd& X, Eigen::VectorXd& y, std::mt19937& gen);
  void SampleXi(Eigen::MatrixXd& X, Eigen::VectorXd& y, std::mt19937& gen);
  void SampleSigma(std::mt19937& gen, double a, double b);
  void SiftGroupIndices(std::vector<int32_t>& group_labels);
  Eigen::VectorXd PredictRandomEffects(Eigen::MatrixXd& X, std::vector<int32_t>& group_labels);
  
 private:
  int num_components_;
  int num_groups_;
  Eigen::MatrixXd W_beta_;
  Eigen::VectorXd alpha_;
  Eigen::MatrixXd xi_;
  Eigen::VectorXd sigma_xi_;
  Eigen::VectorXd sigma_alpha_;
  Eigen::MatrixXd Sigma_xi_;
  Eigen::MatrixXd Sigma_xi_inv_;
  std::vector<std::int32_t> sifted_group_observations_;
  std::vector<std::uint64_t> group_index_begin_;
  std::vector<std::uint64_t> group_index_end_;
  std::vector<std::int32_t> group_index_labels_;
  std::map<int32_t, uint32_t> label_map_;
};

} // namespace StochTree

#endif // STOCHTREE_SAMPLER_H_
