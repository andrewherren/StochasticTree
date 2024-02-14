/*! Copyright (c) 2023 by randtree authors. */

#include <stochtree/cutpoint_candidates.h>
#include <stochtree/data.h>
#include <stochtree/meta.h>
#include <stochtree/model.h>
#include <cmath>
#include <iterator>

namespace StochTree {

XBARTGaussianRegressionModel::XBARTGaussianRegressionModel() {
  config_ = Config();
  alpha_ = config_.alpha;
  beta_ = config_.beta;
  a_sigma_ = config_.a_sigma;
  a_tau_ = config_.a_tau;
  b_sigma_ = config_.b_sigma;
  b_tau_ = config_.b_tau;
  sigma_sq_ = 1;
  tau_ = 1;
  if (config_.random_seed < 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
  } else {
    std::mt19937 gen(config_.random_seed);
  }
}

XBARTGaussianRegressionModel::XBARTGaussianRegressionModel(const Config& config) {
  config_ = config;
  alpha_ = config_.alpha;
  beta_ = config_.beta;
  a_sigma_ = config_.a_sigma;
  a_tau_ = config_.a_tau;
  b_sigma_ = config_.b_sigma;
  b_tau_ = config_.b_tau;
  sigma_sq_ = 1;
  tau_ = 1;
  if (config_.random_seed < 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
  } else {
    std::mt19937 gen(config_.random_seed);
  }
}

void XBARTGaussianRegressionModel::InitializeGlobalParameters(Dataset* dataset) {
  // Compute the outcome mean (used as an offset) and the outcome sd
  double var_y = 0.0;
  double outcome_sum_squares = 0.0;
  double outcome_sum = 0.0;
  double outcome_val;
  data_size_t n = dataset->NumObservations();
  int num_trees = config_.num_trees;
  for (data_size_t i = 0; i < n; i++){
    outcome_val = dataset->OutcomeValue(i);
    outcome_sum += outcome_val;
    outcome_sum_squares += std::pow(outcome_val, 2.0);
  }
  var_y = outcome_sum_squares/n - std::pow(outcome_sum / n, 2.0);
  sd_scale_ = std::sqrt(var_y);
  ybar_offset_ = outcome_sum / n;

  // Scale and center the outcome
  for (data_size_t i = 0; i < n; i++){
    dataset->ResidualSubtract(i, 0, ybar_offset_);
    dataset->ResidualDivide(i, 0, sd_scale_);
  }

  if (config_.data_driven_prior) {
    double var_y = 0.0;
    double outcome_sum_squares = 0.0;
    double outcome_sum = 0.0;
    double outcome_val;
    int num_trees = config_.num_trees;
    for (data_size_t i = 0; i < n; i++){
      outcome_val = dataset->ResidualValue(i);
      outcome_sum += outcome_val;
      outcome_sum_squares += std::pow(outcome_val, 2.0);
    }
    var_y = outcome_sum_squares/n - std::pow(outcome_sum / n, 2.0);
    b_tau_ = 0.5 * var_y / num_trees;
    tau_ = var_y / num_trees;
  }
}

void XBARTGaussianRegressionModel::SampleTree(Dataset* dataset, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                                              SampleNodeMapper* sample_node_mapper, int tree_num) {
  node_t root_id = Tree::kRoot;
  node_t curr_node_id;
  data_size_t curr_node_begin;
  data_size_t curr_node_end;
  data_size_t n = dataset->NumObservations();
  // Reset the mapping from node id to start and end points of sorted indices
  NodeIndexMapReset(n);
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
    SampleSplitRule(dataset, tree, sorted_node_sample_tracker, curr_node_id, curr_node_begin, curr_node_end, split_queue_, sample_node_mapper, tree_num);
  }
}

void XBARTGaussianRegressionModel::SampleLeafParameters(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, Tree* tree) {
  // Vector of leaf indices for tree
  std::vector<node_t> tree_leaves = tree->GetLeaves();
  std::vector<XBARTGaussianRegressionSuffStat> leaf_suff_stats;
  std::normal_distribution<double> leaf_node_dist(0.,1.);
  // Vector of sufficient statistics for each leaf
  for (int i = 0; i < tree_leaves.size(); i++) {
    leaf_suff_stats.push_back(LeafSuffStat(dataset, sorted_node_sample_tracker, tree_leaves[i]));
  }

  // Sample each leaf node parameter
  double node_mean;
  double node_stddev;
  double node_mu;
  for (int i = 0; i < tree_leaves.size(); i++) {
    // Compute posterior mean and variance
    node_mean = LeafPosteriorMean(leaf_suff_stats[i]);
    node_stddev = LeafPosteriorStddev(leaf_suff_stats[i]);
    
    // Draw from N(mean, stddev^2) and set the leaf parameter with each draw
    node_mu = leaf_node_dist(gen)*node_stddev + node_mean;
    // (*tree)[tree_leaves[i]].SetLeaf(node_mu);
    tree->SetLeaf(tree_leaves[i], node_mu);
  }
}

XBARTGaussianRegressionSuffStat XBARTGaussianRegressionModel::LeafSuffStat(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, node_t leaf_id) {
  data_size_t node_begin, node_end;
  std::pair<data_size_t, data_size_t> begin_end;
  if (node_index_map_.find(leaf_id) == node_index_map_.end()) {
    Log::Fatal("Leaf id %d is not present in the tree", leaf_id);
  }
  begin_end = node_index_map_[leaf_id];
  node_begin = begin_end.first;
  node_end = begin_end.second;
  return ComputeNodeSuffStat(dataset, sorted_node_sample_tracker, node_begin, node_end, 0);
}

void XBARTGaussianRegressionModel::SampleGlobalParameters(Dataset* dataset, TreeEnsemble* tree_ensemble, std::set<std::string> update_params) {
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
  
  // Update tau
  if (update_params.count("tau") > 0) {
    // Compute posterior shape and scale parameters for inverse gamma
    double ig_shape_tau = TauPosteriorShape(tree_ensemble);
    double ig_scale_tau = TauPosteriorScale(tree_ensemble);

    double gamma_scale_tau = 1./ig_scale_tau;
    leaf_node_variance_dist_.param(
      std::gamma_distribution<double>::param_type(ig_shape_tau, gamma_scale_tau)
    );
    tau_ = (1/leaf_node_variance_dist_(gen));
  }
}

void XBARTGaussianRegressionModel::SampleSplitRule(Dataset* dataset, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                                                   node_t leaf_node, data_size_t node_begin, data_size_t node_end, std::deque<node_t>& split_queue, 
                                                   SampleNodeMapper* sample_node_mapper, int tree_num) {
  std::vector<double> log_cutpoint_evaluations;
  std::vector<int> cutpoint_features;
  std::vector<double> cutpoint_values;
  std::vector<FeatureType> cutpoint_feature_types;
  StochTree::data_size_t valid_cutpoint_count;
  Cutpoints(dataset, tree, sorted_node_sample_tracker, leaf_node, node_begin, node_end, 
            log_cutpoint_evaluations, cutpoint_features, cutpoint_values, 
            cutpoint_feature_types, valid_cutpoint_count);
  
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
    AddSplitToModel(dataset, tree, sorted_node_sample_tracker, feature_type, leaf_node, node_begin, node_end, 
                    feature_split, split_value, split_queue, sample_node_mapper, tree_num);
  }
}

void XBARTGaussianRegressionModel::Cutpoints(Dataset* dataset, Tree* tree, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                                             node_t leaf_node, data_size_t node_begin, data_size_t node_end, 
                                             std::vector<double>& log_cutpoint_evaluations, 
                                             std::vector<int>& cutpoint_feature, 
                                             std::vector<double>& cutpoint_values, 
                                             std::vector<FeatureType>& cutpoint_feature_types, 
                                             data_size_t& valid_cutpoint_count) {
  // Compute sufficient statistics for the current node
  XBARTGaussianRegressionSuffStat node_suff_stat_ = ComputeNodeSuffStat(dataset, sorted_node_sample_tracker, node_begin, node_end, 0);
  XBARTGaussianRegressionSuffStat left_suff_stat_;
  XBARTGaussianRegressionSuffStat right_suff_stat_;

  // Clear vectors
  log_cutpoint_evaluations.clear();
  cutpoint_feature.clear();
  cutpoint_values.clear();

  // Reset cutpoint grid container
  cutpoint_grid_container.reset(new CutpointGridContainer(dataset, config_));

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
  for (int j = 0; j < dataset->NumCovariates(); j++) {

    // Enumerate cutpoint strides
    cutpoint_grid_container->CalculateStrides(dataset, sorted_node_sample_tracker, leaf_node, node_begin, node_end, j);

    // Iterate through possible cutpoints
    int32_t num_feature_cutpoints = cutpoint_grid_container->NumCutpoints(j);
    feature_type = dataset->GetFeatureType(j);
    // node_row_iter = node_begin;
    ResetSuffStat(left_suff_stat_, 0, 0.0);
    ResetSuffStat(right_suff_stat_, node_suff_stat_.sample_size_, node_suff_stat_.outcome_sum_);
    for (data_size_t cutpoint_idx = 0; cutpoint_idx < (num_feature_cutpoints - 1); cutpoint_idx++) {
      // Unpack cutpoint details, noting that since we partition an entire cutpoint bin to the left, 
      // we must stop one bin before the total number of cutpoint bins
      current_bin_begin = cutpoint_grid_container->BinStartIndex(cutpoint_idx, j);
      current_bin_size = cutpoint_grid_container->BinLength(cutpoint_idx, j);
      next_bin_begin = cutpoint_grid_container->BinStartIndex(cutpoint_idx + 1, j);

      // Accumulate sufficient statistics
      for (data_size_t k = 0; k < current_bin_size; k++) {
        row_iter_idx = current_bin_begin + k;
        feature_sort_idx = sorted_node_sample_tracker->SortIndex(row_iter_idx, j);
        outcome_val = dataset->ResidualValue(feature_sort_idx);
        outcome_val_sq = std::pow(outcome_val, 2.0);
        left_suff_stat_.sample_size_++;
        left_suff_stat_.outcome_sum_ += outcome_val;
        left_suff_stat_.outcome_sum_sq_ += outcome_val_sq;
      }

      // AccumulateRowSuffStat(dataset, left_suff_stat_, node_row_iter, j, node_row_iter);
      right_suff_stat_ = SubtractSuffStat(node_suff_stat_, left_suff_stat_);

      // Store the bin index as the "cutpoint value" - we can use this to query the actual split 
      // value or the set of split categories later on once a split is chose
      cutoff_value = cutpoint_idx;

      // Only include cutpoint for consideration if it defines a valid split in the training data
      valid_split = ((left_suff_stat_.sample_size_ >= config_.min_data_in_leaf) && 
                      (right_suff_stat_.sample_size_ >= config_.min_data_in_leaf));
      if (valid_split) {
        num_cutpoints++;
        // Add to split rule vector
        cutpoint_feature_types.push_back(feature_type);
        cutpoint_feature.push_back(j);
        cutpoint_values.push_back(cutoff_value);
        // Add the log marginal likelihood of the split to the split eval vector 
        log_split_eval = SplitLogMarginalLikelihood(left_suff_stat_, right_suff_stat_);
        log_cutpoint_evaluations.push_back(log_split_eval);
      }
    }
  }

  // Evaluate the "no-split" option
  double log_no_split_eval = 0.;
  int node_depth = tree->GetDepth(leaf_node);
  cutpoint_feature.push_back(-1);
  cutpoint_values.push_back(std::numeric_limits<double>::max());
  log_no_split_eval = NoSplitLogMarginalLikelihood(node_suff_stat_, node_depth, num_cutpoints);
  log_cutpoint_evaluations.push_back(log_no_split_eval);
  valid_cutpoint_count = num_cutpoints;
}

void XBARTGaussianRegressionModel::AddSplitToModel(Dataset* dataset, Tree* tree, SortedNodeSampleTracker* sort_node_sample_tracker, 
                                                   FeatureType feature_type, node_t leaf_node, data_size_t node_begin, data_size_t node_end, 
                                                   int feature_split, double split_value, std::deque<node_t>& split_queue, 
                                                   SampleNodeMapper* sample_node_mapper, int tree_num) {
  // Compute the sufficient statistics for the new left and right node as well as the parent node being split
  XBARTGaussianRegressionSuffStat node_suff_stat_;
  XBARTGaussianRegressionSuffStat left_suff_stat_;
  XBARTGaussianRegressionSuffStat right_suff_stat_;
  ResetSuffStat(left_suff_stat_);
  node_suff_stat_ = ComputeNodeSuffStat(dataset, sort_node_sample_tracker, node_begin, node_end, feature_split);
  
  // Actual numeric cutpoint used for ordered categorical and numeric features
  double split_value_numeric;

  // Split the tree at leaf node
  // Use 0 as a "temporary" leaf value since we sample 
  // all leaf parameters after tree sampling is complete
  double left_leaf_value = 0.;
  double right_leaf_value = 0.;
  if (feature_type == FeatureType::kUnorderedCategorical) {
    // Determine the number of categories available in a categorical split and the set of categories that route observations to the left node after split
    int num_categories;
    std::vector<std::uint32_t> categories = cutpoint_grid_container->CutpointVector(static_cast<std::uint32_t>(split_value), feature_split);

    // Accumulate split rule sufficient statistics
    AccumulateSplitRule(dataset, sort_node_sample_tracker, left_suff_stat_, feature_split, categories, node_begin, node_end, true);
    right_suff_stat_ = SubtractSuffStat(node_suff_stat_, left_suff_stat_);

    // Perform the split
    tree->ExpandNode(leaf_node, feature_split, categories, true, left_leaf_value, right_leaf_value);
    // Partition the dataset according to the new split rule and 
    // determine the beginning and end of the new left and right nodes
    PartitionLeaf(dataset, sort_node_sample_tracker, leaf_node, node_begin, node_suff_stat_.sample_size_, feature_split, categories);
  } else {
    if (feature_type == FeatureType::kOrderedCategorical) {
      // Convert the bin split to an actual split value
      split_value_numeric = cutpoint_grid_container->CutpointValue(static_cast<std::uint32_t>(split_value), feature_split);

      // Accumulate split rule sufficient statistics
      AccumulateSplitRule(dataset, sort_node_sample_tracker, left_suff_stat_, feature_split, split_value_numeric, node_begin, node_end, true);
      right_suff_stat_ = SubtractSuffStat(node_suff_stat_, left_suff_stat_);
      
      tree->ExpandNode(leaf_node, feature_split, split_value_numeric, true, left_leaf_value, right_leaf_value);
      // Partition the dataset according to the new split rule and 
      // determine the beginning and end of the new left and right nodes
      PartitionLeaf(dataset, sort_node_sample_tracker, leaf_node, node_begin, node_suff_stat_.sample_size_, feature_split, split_value_numeric);
    } else if (feature_type == FeatureType::kNumeric) {
      // Convert the bin split to an actual split value
      split_value_numeric = cutpoint_grid_container->CutpointValue(static_cast<std::uint32_t>(split_value), feature_split);

      // Accumulate split rule sufficient statistics
      AccumulateSplitRule(dataset, sort_node_sample_tracker, left_suff_stat_, feature_split, split_value_numeric, node_begin, node_end, true);
      right_suff_stat_ = SubtractSuffStat(node_suff_stat_, left_suff_stat_);
      
      tree->ExpandNode(leaf_node, feature_split, split_value_numeric, true, left_leaf_value, right_leaf_value);
      // Partition the dataset according to the new split rule and 
      // determine the beginning and end of the new left and right nodes
      PartitionLeaf(dataset, sort_node_sample_tracker, leaf_node, node_begin, node_suff_stat_.sample_size_, feature_split, split_value_numeric);
    } else {
      Log::Fatal("Invalid split type");
    }
  }
  node_t left_node = tree->LeftChild(leaf_node);
  node_t right_node = tree->RightChild(leaf_node);

  // Update the leaf node observation tracker
  sort_node_sample_tracker->UpdateObservationMapping(left_node, tree_num, sample_node_mapper);
  sort_node_sample_tracker->UpdateObservationMapping(right_node, tree_num, sample_node_mapper);

  // Add the begin and end indices for the new left and right nodes to node_index_map
  AddNode(left_node, node_begin, node_begin + left_suff_stat_.sample_size_);
  AddNode(right_node, node_begin + left_suff_stat_.sample_size_, node_end);

  // Add the left and right nodes to the split tracker
  split_queue.push_front(right_node);
  split_queue.push_front(left_node);
}

double XBARTGaussianRegressionModel::SplitLogMarginalLikelihood(const XBARTGaussianRegressionSuffStat& left_stat, 
                                                                const XBARTGaussianRegressionSuffStat& right_stat) {
  // Unpack node sufficient statistics
  data_size_t left_n = left_stat.sample_size_;
  data_size_t right_n = right_stat.sample_size_;
  double left_sum_y = left_stat.outcome_sum_;
  double right_sum_y = right_stat.outcome_sum_;
  double sum_y_sq = left_stat.outcome_sum_sq_ + right_stat.outcome_sum_sq_;
  
  // Compute left node contribution to log marginal likelihood
  double left_prior_contrib = 0.5 * std::log(sigma_sq_) - 0.5 * std::log(sigma_sq_ + tau_*left_n);
  double left_data_contrib = 0.5 * (tau_*std::pow(left_sum_y, 2.)) / (sigma_sq_*(sigma_sq_ + tau_*left_n));
  double left_exponent = left_prior_contrib + left_data_contrib;

  // Compute right node contribution to log marginal likelihood
  double right_prior_contrib = 0.5 * std::log(sigma_sq_) - 0.5 * std::log(sigma_sq_ + tau_*right_n);
  double right_data_contrib = 0.5 * (tau_*std::pow(right_sum_y, 2.)) / (sigma_sq_*(sigma_sq_ + tau_*right_n));
  double right_exponent = right_prior_contrib + right_data_contrib;

  // Compute the normalizing components of the log marginal likelihood
  data_size_t n = left_n + right_n;
  double normalizing_component = (-0.5)*(n)*(std::log(sigma_sq_)) - 0.5 * (sum_y_sq/(sigma_sq_));
  
  return normalizing_component + left_exponent + right_exponent;
}

double XBARTGaussianRegressionModel::SplitMarginalLikelihood(const XBARTGaussianRegressionSuffStat& left_stat, 
                                                             const XBARTGaussianRegressionSuffStat& right_stat) {
  return std::exp(SplitLogMarginalLikelihood(left_stat, right_stat));
}

double XBARTGaussianRegressionModel::NoSplitLogMarginalLikelihood(const XBARTGaussianRegressionSuffStat& node_stat, 
                                                                  int node_depth, data_size_t num_split_candidates) {
  // Unpack node sufficient statistics
  data_size_t node_n = node_stat.sample_size_;
  double node_sum_y = node_stat.outcome_sum_;
  double node_sum_y_sq = node_stat.outcome_sum_sq_;

  // Compute leaf prior contribution to log marginal likelihood
  double prior_contrib = std::log(sigma_sq_) - std::log(sigma_sq_ + tau_*node_n);

  // Compute data likelihood contribution to log marginal likelihood
  double data_contrib = ((tau_*std::pow(node_sum_y, 2.0))/(sigma_sq_*(sigma_sq_ + tau_*node_n)));
  
  // Combine the results with prior probability on tree structures
  double exponent = (prior_contrib + data_contrib) * 0.5;

  // BART prior adjustments
  // Handle the case where a node is exactly at min_data_in_leaf so there are no valid split candidates
  // In this case the no split option will be chosen by default, but taking std::log(num_split_candidates) yields -inf
  double bart_prior_no_split_adj;
  if (num_split_candidates == 0) {
    bart_prior_no_split_adj = std::log(((std::pow(1+node_depth, beta_))/alpha_) - 1.0);
  } else {
    bart_prior_no_split_adj = std::log(((std::pow(1+node_depth, beta_))/alpha_) - 1.0) + std::log(num_split_candidates);
  }

  // Compute the normalizing component of the log marginal likelihood
  double normalizing_component = - 0.5 * (node_n)*(std::log(sigma_sq_)) - 0.5 * (node_sum_y_sq/(sigma_sq_));
  
  return exponent + bart_prior_no_split_adj + normalizing_component;
}

double XBARTGaussianRegressionModel::NoSplitMarginalLikelihood(const XBARTGaussianRegressionSuffStat& node_stat, 
                                                               int node_depth, data_size_t num_split_candidates) {
  return std::exp(NoSplitLogMarginalLikelihood(node_stat, node_depth, num_split_candidates));
}

XBARTGaussianRegressionSuffStat XBARTGaussianRegressionModel::ComputeNodeSuffStat(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                                                                                  data_size_t node_begin, data_size_t node_end, int feature_idx) {
  XBARTGaussianRegressionSuffStat suff_stat;
  // Reset sample size and outcome sum
  data_size_t sample_size = 0;
  double outcome;
  double outcome_sum = 0.;
  double outcome_sum_sq = 0.;
  data_size_t sort_idx;
  // Compute the total sufficient statistics for a node
  for (data_size_t i = node_begin; i < node_end; i++) {
    // We could also compute this as node_end - node_begin
    sample_size++;
    // Each feature has different sort indices, but since we just 
    // need a running total sample size and running total outcome 
    // per node we can use any feature's sort indices to get the 
    // corresponding outcome information
    sort_idx = sorted_node_sample_tracker->SortIndex(i, feature_idx);
    outcome = dataset->ResidualValue(sort_idx);
    outcome_sum += outcome;
    outcome_sum_sq += std::pow(outcome, 2.0);
  }
  suff_stat.sample_size_ = sample_size;
  suff_stat.outcome_sum_ = outcome_sum;
  suff_stat.outcome_sum_sq_ = outcome_sum_sq;
  return suff_stat;
}


XBARTGaussianRegressionSuffStat XBARTGaussianRegressionModel::SubtractSuffStat(const XBARTGaussianRegressionSuffStat& first_node_suff_stat, 
                                                                               const XBARTGaussianRegressionSuffStat& second_node_suff_stat) {
  XBARTGaussianRegressionSuffStat suff_stat;
  suff_stat.sample_size_ = first_node_suff_stat.sample_size_ - second_node_suff_stat.sample_size_;
  suff_stat.outcome_sum_ = first_node_suff_stat.outcome_sum_ - second_node_suff_stat.outcome_sum_;
  suff_stat.outcome_sum_sq_ = first_node_suff_stat.outcome_sum_sq_ - second_node_suff_stat.outcome_sum_sq_;
  return suff_stat;
}

void XBARTGaussianRegressionModel::ResetSuffStat(XBARTGaussianRegressionSuffStat& suff_stat, data_size_t sample_size, double outcome_sum, double outcome_sum_sq) {
  suff_stat.sample_size_ = sample_size;
  suff_stat.outcome_sum_ = outcome_sum;
  suff_stat.outcome_sum_sq_ = outcome_sum_sq;
}

void XBARTGaussianRegressionModel::AccumulateSplitRule(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                                                       XBARTGaussianRegressionSuffStat& suff_stat, 
                                                       int split_col, double split_value, data_size_t node_begin, 
                                                       data_size_t node_end, bool is_left) {
  double feature_value;
  double outcome_value;
  double outcome_value_sq;
  suff_stat.sample_size_ = 0;
  suff_stat.outcome_sum_ = 0.0;
  suff_stat.outcome_sum_sq_ = 0.0;
  data_size_t sort_idx;
  bool split_true;
  for (data_size_t i = node_begin; i < node_end; i++) {
    sort_idx = sorted_node_sample_tracker->SortIndex(i, split_col);
    feature_value = dataset->CovariateValue(sort_idx, split_col);
    outcome_value = dataset->ResidualValue(sort_idx, 0);
    outcome_value_sq = std::pow(outcome_value, 2.0);
    split_true = SplitTrueNumeric(feature_value, split_value);
    // Only accumulate sample sufficient statistics if either 
    // (a) the accumulated sufficient statistic is for a left node and the split rule is true, or
    // (b) the accumulated sufficient statistic is for a right node and the split rule is false
    if (split_true && is_left){
      suff_stat.sample_size_++;
      suff_stat.outcome_sum_ += outcome_value;
      suff_stat.outcome_sum_sq_ += outcome_value_sq;
    } else if (!split_true && !is_left) {
      suff_stat.sample_size_++;
      suff_stat.outcome_sum_ += outcome_value;
      suff_stat.outcome_sum_sq_ += outcome_value_sq;
    }
  }
}



void XBARTGaussianRegressionModel::AccumulateSplitRule(Dataset* dataset, SortedNodeSampleTracker* sorted_node_sample_tracker, 
                                                       XBARTGaussianRegressionSuffStat& suff_stat, 
                                                       int split_col, std::vector<std::uint32_t> const& categorical_indices, 
                                                       data_size_t node_begin, data_size_t node_end, bool is_left) {
  double feature_value;
  double outcome_value;
  double outcome_value_sq;
  suff_stat.sample_size_ = 0;
  suff_stat.outcome_sum_ = 0.0;
  suff_stat.outcome_sum_sq_ = 0.0;
  data_size_t sort_idx;
  bool split_true;
  for (data_size_t i = node_begin; i < node_end; i++) {
    sort_idx = sorted_node_sample_tracker->SortIndex(i, split_col);
    feature_value = dataset->CovariateValue(sort_idx, split_col);
    outcome_value = dataset->ResidualValue(sort_idx, 0);
    outcome_value_sq = std::pow(outcome_value, 2.0);
    split_true = SplitTrueCategorical(feature_value, categorical_indices);
    // Only accumulate sample sufficient statistics if either 
    // (a) the accumulated sufficient statistic is for a left node and the split rule is true, or
    // (b) the accumulated sufficient statistic is for a right node and the split rule is false
    if (split_true && is_left){
      suff_stat.sample_size_++;
      suff_stat.outcome_sum_ += outcome_value;
      suff_stat.outcome_sum_sq_ += outcome_value_sq;
    } else if (!split_true && !is_left) {
      suff_stat.sample_size_++;
      suff_stat.outcome_sum_ += outcome_value;
      suff_stat.outcome_sum_sq_ += outcome_value_sq;
    }
  }
}

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
  // [Giacomo] Shouldn't there be 0 = log(1) here instead of 1? Actually, it
  // makes no difference because the check is log U(0, 1) <= log_mh_ratio, so
  // the case with the log ratio above 0 is handled implicitly, since the l.h.s.
  // is always <= 0.
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
  // [Giacomo] redundant
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
      // [Giacomo] use multiplication instead of std::pow
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
