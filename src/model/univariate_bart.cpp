/*! Copyright (c) 2023 by randtree authors. */

#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/meta.h>
#include <stochtree/model.h>
#include <cmath>
#include <iterator>

namespace StochTree {

BARTGaussianRegressionModel::BARTGaussianRegressionModel() {
  config_ = Config();
  nu_ = config_.nu;
  lambda_ = config_.lambda;
  mu_mean_ = config_.mu_mean;
  mu_sigma_ = config_.mu_sigma;
  alpha_ = config_.alpha;
  beta_ = config_.beta;
  sigma_sq_ = 1;
  if (config_.random_seed < 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
  } else {
    std::mt19937 gen(config_.random_seed);
  }
}

BARTGaussianRegressionModel::BARTGaussianRegressionModel(const Config& config) {
  config_ = config;
  nu_ = config_.nu;
  lambda_ = config_.lambda;
  mu_mean_ = config_.mu_mean;
  mu_sigma_ = config_.mu_sigma;
  alpha_ = config_.alpha;
  beta_ = config_.beta;
  sigma_sq_ = 1;
  if (config_.random_seed < 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
  } else {
    std::mt19937 gen(config_.random_seed);
  }
}

void BARTGaussianRegressionModel::InitializeGlobalParameters(Dataset* dataset) {
  // Compute the outcome mean (used as an offset) and the outcome sd
  double var_y = 0.0;
  double outcome_sum_squares = 0.0;
  double outcome_sum = 0.0;
  double outcome_val;
  // data_size_t n = train_data->num_data();
  data_size_t n = dataset->NumObservations();
  int num_trees = config_.num_trees;
  for (data_size_t i = 0; i < n; i++){
    // outcome_val = train_data->get_outcome_value(i);
    outcome_val = dataset->OutcomeValue(i, 0);
    outcome_sum += outcome_val;
    outcome_sum_squares += std::pow(outcome_val, 2.0);
  }
  var_y = outcome_sum_squares/n - std::pow(outcome_sum / n, 2.0);
  sd_scale_ = std::sqrt(var_y);
  ybar_offset_ = outcome_sum / n;

  // Scale and center the outcome
  for (data_size_t i = 0; i < n; i++){
    // outcome_val = train_data->get_outcome_value(i);
    dataset->ResidualSubtract(i, 0, ybar_offset_);
    dataset->ResidualDivide(i, 0, sd_scale_);
  }

  // Calibrate priors if called for
  if (config_.data_driven_prior) {
    double var_y = 0.0;
    double sd_y = 0.0;
    double y_min = std::numeric_limits<double>::max();
    double y_max = std::numeric_limits<double>::min();
    double outcome_sum_squares = 0.0;
    double outcome_sum = 0.0;
    double outcome_val;
    double ybar;
    // data_size_t n = train_data->num_data();
    data_size_t n = dataset->NumObservations();
    int num_trees = config_.num_trees;
    // Compute min and max
    for (data_size_t i = 0; i < n; i++) {
      // outcome_val = train_data->get_residual_value(i);
      outcome_val = dataset->ResidualValue(i, 0);
      outcome_sum += outcome_val;
      outcome_sum_squares += std::pow(outcome_val, 2.0);
      if (outcome_val < y_min) {
        y_min = outcome_val;
      }
      if (outcome_val > y_max) {
        y_max = outcome_val;
      }
    }
    var_y = outcome_sum_squares/n - std::pow(outcome_sum / n, 2.0);
    sd_y = std::sqrt(var_y);
    ybar = outcome_sum / n;

    // Calibrate lambda so that p(sigma^2 < var_y) = 0.95
    lambda_ = var_y*(2/nu_)*boost::math::gamma_p_inv(nu_/2, 0.95);

    // Calibrate mu_mean and mu_sigma
    mu_mean_ = (y_max + y_min)/(2*config_.num_trees);
    mu_sigma_ = (y_max - y_min)/(2*1.96*std::sqrt(config_.num_trees));
  }

  // Set sigma_sq equal to its prior mean
  sigma_sq_ = (nu_*lambda_) / (nu_ - 2.0);
}

void BARTGaussianRegressionModel::SampleTree(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_tracker,
                                             SampleNodeMapper* sample_node_mapper, int tree_num) {
  // Perform one MCMC step
  MCMCTreeStep(dataset, tree, node_tracker, tree_num);

  // Update the SampleNodeMapper
  node_tracker->UpdateObservationMapping(tree, tree_num, sample_node_mapper);
}

void BARTGaussianRegressionModel::MCMCTreeStep(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int tree_num) {
  // Compute sufficient statistics for each node
  BARTGaussianRegressionSuffStat suff_stat;
  node_suff_stats_.clear();
  for (int i = 0; i < tree->NumNodes(); i++) {
    node_suff_stats_.push_back(ComputeNodeSufficientStatistics(dataset, tree, node_tracker, i, tree_num));
  }

  // Determine whether it is possible to grow any of the leaves
  bool grow_possible = false;
  std::vector<int> leaves = tree->GetLeaves();
  for (auto& leaf: leaves) {
    if (node_suff_stats_[leaf].sample_size_ > 2*config_.min_data_in_leaf) {
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
  std::vector<double> step_probs(2);
  if (grow_possible && prune_possible) {
    step_probs = {0.5, 0.5};
  } else if (!grow_possible && prune_possible) {
    step_probs = {0.0, 1.0};
  } else if (grow_possible && !prune_possible) {
    step_probs = {1.0, 0.0};
  } else {
    Log::Fatal("In this tree, neither grow nor prune is possible");
  }
  std::discrete_distribution<> step_dist(step_probs.begin(), step_probs.end());

  // Draw a split rule at random
  data_size_t step_chosen = step_dist(gen);
  bool accept;
  
  if (step_chosen == 0) {
    GrowMCMC(dataset, tree, node_tracker, accept, tree_num);
  } else {
    PruneMCMC(dataset, tree, node_tracker, accept, tree_num);
  }
}

void BARTGaussianRegressionModel::GrowMCMC(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_tracker, bool& accept, int tree_num) {
  // Choose a leaf node at random
  int num_leaves = tree->NumLeaves();
  std::vector<int> leaves = tree->GetLeaves();
  std::vector<double> leaf_weights(num_leaves);
  std::fill(leaf_weights.begin(), leaf_weights.end(), 1.0/num_leaves);
  std::discrete_distribution<> leaf_dist(leaf_weights.begin(), leaf_weights.end());
  int leaf_chosen = leaves[leaf_dist(gen)];
  int leaf_depth = tree->GetDepth(leaf_chosen);

  // Select a split variable at random
  int p = dataset->NumCovariates();
  std::vector<double> var_weights(p);
  std::fill(var_weights.begin(), var_weights.end(), 1.0/p);
  std::discrete_distribution<> var_dist(var_weights.begin(), var_weights.end());
  int var_chosen = var_dist(gen);

  // Determine the range of possible cutpoints
  double var_min, var_max;
  VarSplitRange(dataset, tree, node_tracker, leaf_chosen, var_chosen, var_min, var_max, tree_num);
  if (var_max <= var_min) {
    accept = true;
    return;
  }
  // Split based on var_min to var_max in a given node
  std::uniform_real_distribution<double> split_point_dist(var_min, var_max);
  double split_point_chosen = split_point_dist(gen);

  // Compute sufficient statistics of the two new nodes
  BARTGaussianRegressionSuffStat left_suff_stat = {0, 0.0, 0.0};
  BARTGaussianRegressionSuffStat right_suff_stat = {0, 0.0, 0.0};
  ComputeSplitSuffStats(dataset, tree, node_tracker, leaf_chosen,
                        left_suff_stat, right_suff_stat,
                        var_chosen, split_point_chosen, tree_num);
  
  // Retrieve sufficient statistic for split node
  BARTGaussianRegressionSuffStat node_suff_stat = node_suff_stats_[leaf_chosen];

  // Compute the marginal likelihood
  double split_likelihood = SplitMarginalLikelihood(left_suff_stat, right_suff_stat);
  double no_split_likelihood = NoSplitMarginalLikelihood(node_suff_stat);
  
  // Determine probability of growing the split node and its two new left and right nodes
  double pg = alpha_ * std::pow(1+leaf_depth, -beta_);
  double pgl = alpha_ * std::pow(1+leaf_depth+1, -beta_);
  double pgr = alpha_ * std::pow(1+leaf_depth+1, -beta_);

  // Determine whether a "grow" move is possible from the newly formed tree
  // in order to compute the probability of choosing "prune" from the new tree
  // (which is always possible by construction)
  bool non_constant = NodesNonConstantAfterSplit(dataset, tree, node_tracker, leaf_chosen, var_chosen, split_point_chosen, tree_num);
  bool min_samples_left_check = left_suff_stat.sample_size_ > 2*config_.min_data_in_leaf;
  bool min_samples_right_check = right_suff_stat.sample_size_ > 2*config_.min_data_in_leaf;
  double prob_prune_new;
  if (non_constant && min_samples_left_check && min_samples_right_check) {
    prob_prune_new = 0.5;
  } else {
    prob_prune_new = 1.0;
  }
  double prob_grow_old = 0.5;

  // Determine the number of leaves in the current tree and leaf parents in the proposed tree
  int num_leaf_parents = tree->NumLeafParents();
  double p_leaf = 1/num_leaves;
  double p_leaf_parent = 1/(num_leaf_parents+1);

  // Compute the final MH ratio
  double log_mh_ratio = (
    std::log(pg) + std::log(1-pgl) + std::log(1-pgr) - std::log(1-pg) + std::log(prob_prune_new) +
    std::log(p_leaf_parent) - std::log(prob_grow_old) - std::log(p_leaf) + std::log(split_likelihood) - std::log(no_split_likelihood)
  );
  // Threshold at 0
  if (log_mh_ratio > 1) {
    log_mh_ratio = 1;
  }

  // Draw a uniform random variable and accept/reject the proposal on this basis
  std::uniform_real_distribution<double> mh_accept(0.0, 1.0);
  double log_acceptance_prob = std::log(mh_accept(gen));
  if (log_acceptance_prob <= log_mh_ratio) {
    accept = true;
    AddSplitToModel(dataset, tree, node_tracker, leaf_chosen, var_chosen, split_point_chosen, node_suff_stat, left_suff_stat, right_suff_stat, tree_num);
  } else {
    accept = false;
  }
}

void BARTGaussianRegressionModel::PruneMCMC(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_tracker, bool& accept, int tree_num) {
  // Choose a "leaf parent" node at random
  int num_leaves = tree->NumLeaves();
  int num_leaf_parents = tree->NumLeafParents();
  std::vector<int> leaf_parents = tree->GetLeafParents();
  std::vector<double> leaf_parent_weights(num_leaf_parents);
  std::fill(leaf_parent_weights.begin(), leaf_parent_weights.end(), 1.0/num_leaf_parents);
  std::discrete_distribution<> leaf_parent_dist(leaf_parent_weights.begin(), leaf_parent_weights.end());
  int leaf_parent_chosen = leaf_parents[leaf_parent_dist(gen)];
  int leaf_parent_depth = tree->GetDepth(leaf_parent_chosen);
  // int left_node = (*tree)[leaf_parent_chosen].LeftChild();
  // int right_node = (*tree)[leaf_parent_chosen].RightChild();
  // int feature_split = (*tree)[leaf_parent_chosen].SplitIndex();
  // double split_value = (*tree)[leaf_parent_chosen].NumericSplitCond();
  int left_node = tree->LeftChild(leaf_parent_chosen);
  int right_node = tree->RightChild(leaf_parent_chosen);
  int feature_split = tree->SplitIndex(leaf_parent_chosen);
  double split_value = tree->Threshold(leaf_parent_chosen);

  // Retrieve sufficient statistic for split node
  BARTGaussianRegressionSuffStat node_suff_stat = node_suff_stats_[leaf_parent_chosen];

  // Retrieve sufficient statistic for its left and right nodes
  BARTGaussianRegressionSuffStat left_suff_stat = node_suff_stats_[left_node];
  BARTGaussianRegressionSuffStat right_suff_stat = node_suff_stats_[right_node];

  // Compute the marginal likelihood
  double split_likelihood = SplitMarginalLikelihood(left_suff_stat, right_suff_stat);
  double no_split_likelihood = NoSplitMarginalLikelihood(node_suff_stat);
  
  // Determine probability of growing the split node and its two new left and right nodes
  double pg = alpha_ * std::pow(1+leaf_parent_depth, -beta_);
  double pgl = alpha_ * std::pow(1+leaf_parent_depth+1, -beta_);
  double pgr = alpha_ * std::pow(1+leaf_parent_depth+1, -beta_);

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
  bool non_constant_left = NodeNonConstant(dataset, tree, node_tracker, left_node, tree_num);
  bool non_constant_right = NodeNonConstant(dataset, tree, node_tracker, right_node, tree_num);
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
    std::log(p_leaf) - std::log(prob_grow_new) - std::log(p_leaf_parent) + std::log(no_split_likelihood) - std::log(split_likelihood)
  );
  // Threshold at 0
  if (log_mh_ratio > 0) {
    log_mh_ratio = 0;
  }

  // Draw a uniform random variable and accept/reject the proposal on this basis
  std::uniform_real_distribution<double> mh_accept(0.0, 1.0);
  double log_acceptance_prob = std::log(mh_accept(gen));
  if (log_acceptance_prob <= log_mh_ratio) {
    accept = true;
    RemoveSplitFromModel(dataset, tree, node_tracker, leaf_parent_chosen, left_node, right_node, feature_split, split_value, node_suff_stat, left_suff_stat, right_suff_stat, tree_num);
  } else {
    accept = false;
  }
}

BARTGaussianRegressionSuffStat BARTGaussianRegressionModel::ComputeNodeSufficientStatistics(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int node_id, int tree_num) {
  data_size_t idx;
  double outcome_value;
  BARTGaussianRegressionSuffStat node_suff_stat{0, 0.0, 0.0};
  auto tree_node_tracker = node_tracker->GetFeaturePartition(tree_num);
  auto node_begin_iter = tree_node_tracker->indices_.begin() + tree_node_tracker->NodeBegin(node_id);
  auto node_end_iter = tree_node_tracker->indices_.begin() + tree_node_tracker->NodeEnd(node_id);
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    idx = *i;
    outcome_value = dataset->ResidualValue(idx, 0);
    node_suff_stat.sample_size_++;
    node_suff_stat.outcome_sum_ += outcome_value;
    node_suff_stat.outcome_sum_sq_ += std::pow(outcome_value, 2.0);
  }
  return node_suff_stat;
}

bool BARTGaussianRegressionModel::NodeNonConstant(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_tracker, int node_id, int tree_num) {
  int p = dataset->NumCovariates();
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
      feature_value = dataset->CovariateValue(idx, j);
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

bool BARTGaussianRegressionModel::NodesNonConstantAfterSplit(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_tracker,
                                                             int leaf_split, int feature_split, double split_value, int tree_num) {
  int p = dataset->NumCovariates();
  data_size_t idx;
  double feature_value;
  double split_feature_value;
  double var_max_left;
  double var_min_left;
  double var_max_right;
  double var_min_right;
  auto tree_node_tracker = node_tracker->GetFeaturePartition(tree_num);
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
      feature_value = dataset->CovariateValue(idx, j);
      split_feature_value = dataset->CovariateValue(idx, feature_split);
      if (split_feature_value <= split_value) {
        if (var_max_left < feature_value) {
          var_max_left = feature_value;
        } else if (var_min_left > feature_value) {
          var_max_left = feature_value;
        }
      } else {
        if (var_max_right < feature_value) {
          var_max_right = feature_value;
        } else if (var_min_right > feature_value) {
          var_max_right = feature_value;
        }
      }
    }
    if ((var_max_left > var_min_left) && (var_max_right > var_min_right)) {
      return true;
    }
  }
  return false;
}

void BARTGaussianRegressionModel::SampleLeafParameters(Dataset* dataset, Tree* tree) {
  // Vector of leaf indices for tree
  std::vector<node_t> tree_leaves = tree->GetLeaves();
  // Standard normal distribution
  std::normal_distribution<double> leaf_node_dist(0.,1.);

  // Sample each leaf node parameter
  double node_mean;
  double node_stddev;
  double node_mu;
  for (int i = 0; i < tree_leaves.size(); i++) {
    // Compute posterior mean and variance
    node_mean = LeafPosteriorMean(node_suff_stats_[tree_leaves[i]]);
    node_stddev = LeafPosteriorStddev(node_suff_stats_[tree_leaves[i]]);
    
    // Draw from N(mean, stddev^2) and set the leaf parameter with each draw
    node_mu = leaf_node_dist(gen)*node_stddev + node_mean;
    // (*tree)[tree_leaves[i]].SetLeaf(node_mu);
    tree->SetLeaf(tree_leaves[i], node_mu);
  }
}

void BARTGaussianRegressionModel::ComputeSplitSuffStats(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_tracker,
                                                        int leaf_split, BARTGaussianRegressionSuffStat& left_suff_stat,
                                                        BARTGaussianRegressionSuffStat& right_suff_stat,
                                                        int feature_split, double split_value, int tree_num) {
  double outcome_value;
  double feature_value;
  auto tree_node_tracker = node_tracker->GetFeaturePartition(tree_num);
  data_size_t node_begin = tree_node_tracker->NodeBegin(leaf_split);
  data_size_t node_end = tree_node_tracker->NodeEnd(leaf_split);
  data_size_t idx;
  auto node_begin_iter = tree_node_tracker->indices_.begin() + node_begin;
  auto node_end_iter = tree_node_tracker->indices_.begin() + node_end;
  for (auto i = node_begin_iter; i != node_end_iter; i++) {
    idx = *i;
    feature_value = dataset->CovariateValue(idx, feature_split);
    outcome_value = dataset->ResidualValue(idx);
    if (feature_value <= split_value) {
      left_suff_stat.sample_size_++;
      left_suff_stat.outcome_sum_ += outcome_value;
      left_suff_stat.outcome_sum_sq_ += std::pow(outcome_value, 2.0);
    } else {
      right_suff_stat.sample_size_++;
      right_suff_stat.outcome_sum_ += outcome_value;
      right_suff_stat.outcome_sum_sq_ += std::pow(outcome_value, 2.0);
    }
  }
}

void BARTGaussianRegressionModel::VarSplitRange(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_tracker,
                                                int leaf_split, int feature_split, double& var_min, double& var_max, int tree_num) {
  data_size_t n = dataset->NumObservations();
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
    feature_value = dataset->CovariateValue(idx, feature_split);
    if (feature_value < var_min) {
      var_min = feature_value;
    } else if (feature_value > var_max) {
      var_max = feature_value;
    }
  }
}

void BARTGaussianRegressionModel::SampleGlobalParameters(Dataset* dataset, TreeEnsemble* tree_ensemble, std::set<std::string> update_params) {
  // Update sigma^2
  if (update_params.count("sigma_sq") > 0) {
    // Compute posterior shape and scale parameters for inverse gamma
    double ig_shape_sig = SigmaPosteriorShape(dataset);
    double ig_scale_sig = SigmaPosteriorScale(dataset);
    
    // C++ standard library provides a gamma distribution with scale
    // parameter, but the correspondence between gamma and IG is that
    // 1 / gamma(a,b) ~ IG(a,b) when b is a __rate__ parameter.
    // Before sampling, we convert ig_scale to a gamma scale parameter by
    // taking its multiplicative inverse.
    double gamma_scale_sig = 1./ig_scale_sig;
    residual_variance_dist_.param(
      std::gamma_distribution<double>::param_type(ig_shape_sig, gamma_scale_sig)
    );
    sigma_sq_ = (1/residual_variance_dist_(gen));
  }
}

void BARTGaussianRegressionModel::AddSplitToModel(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_tracker, node_t leaf_node,
                                                  int feature_split, double split_value, BARTGaussianRegressionSuffStat& node_suff_stat,
                                                  BARTGaussianRegressionSuffStat& left_suff_stat, BARTGaussianRegressionSuffStat& right_suff_stat, int tree_num) {
  // Split the tree at leaf node
  // Use the average outcome in each leaf as a "temporary" leaf value since we sample
  // all leaf parameters after tree sampling is complete
  double left_leaf_value = left_suff_stat.outcome_sum_/left_suff_stat.sample_size_;
  double right_leaf_value = right_suff_stat.outcome_sum_/right_suff_stat.sample_size_;
  // tree->ExpandNode(leaf_node, feature_split, split_value, true, 1., left_leaf_value, right_leaf_value, -1);
  tree->ExpandNode(leaf_node, feature_split, split_value, true, left_leaf_value, right_leaf_value);
  node_t left_node = tree->LeftChild(leaf_node);
  node_t right_node = tree->RightChild(leaf_node);

  // Update the UnsortedNodeSampleTracker
  node_tracker->PartitionTreeNode(dataset, tree_num, leaf_node, left_node, right_node, feature_split, split_value);

  // Update the vector of node sufficient statistics
  if (right_node >= node_suff_stats_.size()) {
    node_suff_stats_.resize(right_node + 1);
  }
  node_suff_stats_.at(left_node) = left_suff_stat;
  node_suff_stats_.at(right_node) = right_suff_stat;
}

void BARTGaussianRegressionModel::RemoveSplitFromModel(Dataset* dataset, Tree* tree, UnsortedNodeSampleTracker* node_tracker, node_t leaf_node, node_t left_node,
                                                       node_t right_node, int feature_split, double split_value, BARTGaussianRegressionSuffStat& node_suff_stat,
                                                       BARTGaussianRegressionSuffStat& left_suff_stat, BARTGaussianRegressionSuffStat& right_suff_stat, int tree_num) {
  // Prune the tree
  double leaf_value = node_suff_stat.outcome_sum_/node_suff_stat.sample_size_;
  tree->ChangeToLeaf(leaf_node, leaf_value);

  // Update the UnsortedNodeSampleTracker
  node_tracker->PruneTreeNodeToLeaf(tree_num, leaf_node);
}

double BARTGaussianRegressionModel::SplitLogMarginalLikelihood(const BARTGaussianRegressionSuffStat& left_stat,
                                                               const BARTGaussianRegressionSuffStat& right_stat) {
  return std::log(SplitMarginalLikelihood(left_stat, right_stat));
}

double BARTGaussianRegressionModel::SplitMarginalLikelihood(const BARTGaussianRegressionSuffStat& left_stat,
                                                            const BARTGaussianRegressionSuffStat& right_stat) {
  // Unpack node sufficient statistics
  data_size_t left_n = left_stat.sample_size_;
  data_size_t right_n = right_stat.sample_size_;
  double left_sum_y = left_stat.outcome_sum_;
  double right_sum_y = right_stat.outcome_sum_;
  
  // Compute left node contribution to marginal likelihood
  double left_prior_contrib = std::sqrt(sigma_sq_ / (sigma_sq_ + mu_sigma_*left_n));
  double left_data_contrib = std::exp(std::pow(sigma_sq_*mu_mean_ + mu_sigma_*left_sum_y, 2.0) / ((2*(sigma_sq_*mu_sigma_))*(sigma_sq_ + mu_sigma_*left_n)));
  double left_exponent = left_prior_contrib * left_data_contrib;

  // Compute right node contribution to log marginal likelihood
  double right_prior_contrib = std::sqrt(sigma_sq_ / (sigma_sq_ + mu_sigma_*right_n));
  double right_data_contrib = std::exp(std::pow(sigma_sq_*mu_mean_ + mu_sigma_*right_sum_y, 2.0) / ((2*(sigma_sq_*mu_sigma_))*(sigma_sq_ + mu_sigma_*right_n)));
  double right_exponent = right_prior_contrib * right_data_contrib;
  
  return left_exponent * right_exponent;
}

double BARTGaussianRegressionModel::NoSplitLogMarginalLikelihood(const BARTGaussianRegressionSuffStat& node_stat) {
  return std::log(NoSplitMarginalLikelihood(node_stat));
}

double BARTGaussianRegressionModel::NoSplitMarginalLikelihood(const BARTGaussianRegressionSuffStat& node_stat) {
  // Unpack node sufficient statistics
  data_size_t n = node_stat.sample_size_;
  double sum_y = node_stat.outcome_sum_;
  
  // Compute non-split node contribution to marginal likelihood
  double prior_contrib = std::sqrt(sigma_sq_ / (sigma_sq_ + mu_sigma_*n));
  double data_contrib = std::exp(std::pow(sigma_sq_*mu_mean_ + mu_sigma_*sum_y, 2.0) / ((2*(sigma_sq_*mu_sigma_))*(sigma_sq_ + mu_sigma_*n)));
  double exponent = prior_contrib * data_contrib;

  return exponent;
}

} // namespace StochTree
