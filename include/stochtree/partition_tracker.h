/*!
 * Copyright (c) 2023 stochtree authors.
 * 
 * Data structures used for tracking dataset through the tree building process.
 * 
 * The first category of data structure tracks observations available in nodes of a tree.
 *   a. UnsortedNodeSampleTracker tracks the observations available in every leaf of an ensemble, 
 *      in no feature-specific sort order. It is primarily designed for use in BART-based algorithms.
 *   b. SortedNodeSampleTracker tracks the observations available in a every leaf of a tree, pre-sorted 
 *      separately for each feature. It is primarily design for use in XBART-based algorithms.
 * 
 * The second category, SampleNodeMapper, maps observations from a dataset to leaf nodes.
 * 
 * SampleNodeMapper is inspired by the design of the DataPartition class in LightGBM, 
 * released under the MIT license with the following copyright:
 * 
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * 
 * SortedNodeSampleTracker is inspired by the "approximate" split finding method in xgboost, released 
 * under the Apache license with the following copyright:
 * 
 * Copyright 2015~2023 by XGBoost Contributors
 */
#ifndef STOCHTREE_NODE_SAMPLE_TRACKER_H_
#define STOCHTREE_NODE_SAMPLE_TRACKER_H_

#include <stochtree/config.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/log.h>
#include <stochtree/tree.h>

#include <cmath>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace StochTree {

/*! \brief Class storing sample-node map for each tree in an ensemble */
class SampleNodeMapper {
 public:
  SampleNodeMapper(int num_trees, data_size_t num_observations) {
    num_trees_ = num_trees;
    num_observations_ = num_observations;
    // Initialize the vector of vectors of leaf indices for each tree
    tree_observation_indices_.resize(num_trees_);
    for (int j = 0; j < num_trees_; j++) {
      tree_observation_indices_[j].resize(num_observations_);
    }
  }
  
  SampleNodeMapper(SampleNodeMapper& other){
    num_trees_ = other.NumTrees();
    num_observations_ = other.NumObservations();
    // Initialize the vector of vectors of leaf indices for each tree
    tree_observation_indices_.resize(num_trees_);
    for (int j = 0; j < num_trees_; j++) {
      tree_observation_indices_[j].resize(num_observations_);
    }
  }

  inline data_size_t GetNodeId(data_size_t sample_id, int tree_id) {
    CHECK_LT(sample_id, num_observations_);
    CHECK_LT(tree_id, num_trees_);
    return tree_observation_indices_[tree_id][sample_id];
  }

  inline void SetNodeId(data_size_t sample_id, int tree_id, int node_id) {
    CHECK_LT(sample_id, num_observations_);
    CHECK_LT(tree_id, num_trees_);
    tree_observation_indices_[tree_id][sample_id] = node_id;
  }
  
  inline int NumTrees() {return num_trees_;}
  
  inline int NumObservations() {return num_observations_;}

  inline void AssignAllSamplesToRoot(int tree_id) {
    for (data_size_t i = 0; i < num_observations_; i++) {
      tree_observation_indices_[tree_id][i] = 0;
    }
  }

 private:
  std::vector<std::vector<int>> tree_observation_indices_;
  int num_trees_;
  data_size_t num_observations_;
};

/*! \brief Mapping nodes to the indices they contain */
class FeatureUnsortedPartition {
 public:
  FeatureUnsortedPartition(data_size_t n);

  /*! \brief Partition a node based on a new split rule */
  void PartitionNode(Dataset* dataset, int node_id, int left_node_id, int right_node_id, int feature_split, double split_value);

  /*! \brief Partition a node based on a new split rule */
  void PartitionNode(Dataset* dataset, int node_id, int left_node_id, int right_node_id, int feature_split, std::vector<std::uint32_t> const& category_list);

  /*! \brief Convert a (currently split) node to a leaf */
  void PruneNodeToLeaf(int node_id);

  /*! \brief Whether node_id is a leaf */
  bool IsLeaf(int node_id);

  /*! \brief Whether node_id is a valid node */
  bool IsValidNode(int node_id);

  /*! \brief Whether node_id's left child is a leaf */
  bool LeftNodeIsLeaf(int node_id);

  /*! \brief Whether node_id's right child is a leaf */
  bool RightNodeIsLeaf(int node_id);

  /*! \brief First index of data points contained in node_id */
  data_size_t NodeBegin(int node_id);

  /*! \brief One past the last index of data points contained in node_id */
  data_size_t NodeEnd(int node_id);

  /*! \brief Parent node_id */
  int Parent(int node_id);

  /*! \brief Left child of node_id */
  int LeftNode(int node_id);

  /*! \brief Right child of node_id */
  int RightNode(int node_id);

  /*! \brief Data indices */
  std::vector<data_size_t> indices_;

  /*! \brief Data indices for a given node */
  std::vector<data_size_t> NodeIndices(int node_id);

  /*! \brief Update SampleNodeMapper for all the observations in node_id */
  void UpdateObservationMapping(int node_id, int tree_id, SampleNodeMapper* sample_node_mapper);

 private:
  // Vectors tracking indices in each node
  std::vector<data_size_t> node_begin_;
  std::vector<data_size_t> node_length_;
  std::vector<int32_t> parent_nodes_;
  std::vector<int32_t> left_nodes_;
  std::vector<int32_t> right_nodes_;
  int num_nodes_, num_deleted_nodes_;
  std::vector<int> deleted_nodes_;

  // Private helper functions
  void ExpandNodeTrackingVectors(int node_id, int left_node_id, int right_node_id, data_size_t node_start_idx, data_size_t num_left, data_size_t num_right);
  void ConvertLeafParentToLeaf(int node_id);
};

/*! \brief Mapping nodes to the indices they contain */
class UnsortedNodeSampleTracker {
 public:
  UnsortedNodeSampleTracker(data_size_t n, int num_trees) {
    feature_partitions_.resize(num_trees);
    num_trees_ = num_trees;
    for (int i = 0; i < num_trees; i++) {
      feature_partitions_[i].reset(new FeatureUnsortedPartition(n));
    }
  }

  /*! \brief Partition a node based on a new split rule */
  void PartitionTreeNode(Dataset* dataset, int tree_id, int node_id, int left_node_id, int right_node_id, int feature_split, double split_value) {
    return feature_partitions_[tree_id]->PartitionNode(dataset, node_id, left_node_id, right_node_id, feature_split, split_value);
  }

  /*! \brief Partition a node based on a new split rule */
  void PartitionTreeNode(Dataset* dataset, int tree_id, int node_id, int left_node_id, int right_node_id, int feature_split, std::vector<std::uint32_t> const& category_list) {
    return feature_partitions_[tree_id]->PartitionNode(dataset, node_id, left_node_id, right_node_id, feature_split, category_list);
  }

  /*! \brief Convert a (currently split) node to a leaf */
  void PruneTreeNodeToLeaf(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->PruneNodeToLeaf(node_id);
  }

  /*! \brief Whether node_id is a leaf */
  bool IsLeaf(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->IsLeaf(node_id);
  }

  /*! \brief Whether node_id is a valid node */
  bool IsValidNode(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->IsValidNode(node_id);
  }

  /*! \brief Whether node_id's left child is a leaf */
  bool LeftNodeIsLeaf(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->LeftNodeIsLeaf(node_id);
  }

  /*! \brief Whether node_id's right child is a leaf */
  bool RightNodeIsLeaf(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->RightNodeIsLeaf(node_id);
  }

  /*! \brief First index of data points contained in node_id */
  data_size_t NodeBegin(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->NodeBegin(node_id);
  }

  /*! \brief One past the last index of data points contained in node_id */
  data_size_t NodeEnd(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->NodeEnd(node_id);
  }

  /*! \brief Parent node_id */
  int Parent(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->Parent(node_id);
  }

  /*! \brief Left child of node_id */
  int LeftNode(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->LeftNode(node_id);
  }

  /*! \brief Right child of node_id */
  int RightNode(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->RightNode(node_id);
  }

  /*! \brief Data indices for a given node */
  std::vector<data_size_t> TreeNodeIndices(int tree_id, int node_id) {
    return feature_partitions_[tree_id]->NodeIndices(node_id);
  }

  /*! \brief Update SampleNodeMapper for all the observations in node_id */
  void UpdateObservationMapping(int node_id, int tree_id, SampleNodeMapper* sample_node_mapper) {
    feature_partitions_[tree_id]->UpdateObservationMapping(node_id, tree_id, sample_node_mapper);
  }

  /*! \brief Update SampleNodeMapper for all the observations in tree */
  void UpdateObservationMapping(Tree* tree, int tree_id, SampleNodeMapper* sample_node_mapper) {
    std::vector<int> leaves = tree->GetLeaves();
    int leaf;
    for (int i = 0; i < leaves.size(); i++) {
      leaf = leaves[i];
      UpdateObservationMapping(leaf, tree_id, sample_node_mapper);
    }
  }

  /*! \brief Number of trees */
  int NumTrees() { return num_trees_; }

  /*! \brief Number of trees */
  FeatureUnsortedPartition* GetFeaturePartition(int i) { return feature_partitions_[i].get(); }

 private:
  // Vectors of feature partitions
  std::vector<std::unique_ptr<FeatureUnsortedPartition>> feature_partitions_;
  int num_trees_;
};

/*! \brief Tracking cutpoints available at a given node */
class NodeOffsetSize {
 public:
  NodeOffsetSize(data_size_t node_offset, data_size_t node_size) : node_begin_{node_offset}, node_size_{node_size}, presorted_{false} {
    node_end_ = node_begin_ + node_size_;
  }

  ~NodeOffsetSize() {}

  void SetSorted() {presorted_ = true;}

  bool IsSorted() {return presorted_;}

  data_size_t Begin() {return node_begin_;}

  data_size_t End() {return node_end_;}

  data_size_t Size() {return node_size_;}

 private:
  data_size_t node_begin_;
  data_size_t node_size_;
  data_size_t node_end_;
  bool presorted_;
};

/*! \brief Forward declaration of partition-based presort tracker */
class FeaturePresortPartition;

/*! \brief Data structure for presorting a feature by its values
 * 
 *  This class is intended to be run *once* on a dataset as it 
 *  pre-sorts each feature across the entire dataset.
 *  
 *  FeaturePresortPartition is intended for use in recursive construction
 *  of new trees, and each new tree's FeaturePresortPartition is initialized 
 *  from a FeaturePresortRoot class so that features are only arg-sorted one time.
 */
class FeaturePresortRoot {
 friend FeaturePresortPartition; 
 public:
  FeaturePresortRoot(Dataset* dataset, int32_t feature_index, FeatureType feature_type) {
    feature_index_ = feature_index;
    ArgsortRoot(dataset);
  }

  ~FeaturePresortRoot() {}

  void ArgsortRoot(Dataset* dataset) {
    data_size_t num_obs = dataset->NumObservations();
    
    // Make a vector of indices from 0 to num_obs - 1
    if (feature_sort_indices_.size() != num_obs){
      feature_sort_indices_.resize(num_obs, 0);
    }
    std::iota(feature_sort_indices_.begin(), feature_sort_indices_.end(), 0);

    // Define a custom comparator to be used with stable_sort:
    // For every two indices l and r store as elements of `data_sort_indices_`, 
    // compare them for sorting purposes by indexing the covariate's raw data with both l and r
    auto comp_op = [&](size_t const &l, size_t const &r) { return std::less<double>{}(dataset->CovariateValue(l, feature_index_), dataset->CovariateValue(r, feature_index_)); };
    std::stable_sort(feature_sort_indices_.begin(), feature_sort_indices_.end(), comp_op);
  }

 private:
  std::vector<data_size_t> feature_sort_indices_;
  int32_t feature_index_;
};

/*! \brief Container class for FeaturePresortRoot objects stored for every feature in a dataset */
class FeaturePresortRootContainer {
 public:
  FeaturePresortRootContainer(Dataset* dataset) {
    num_features_ = dataset->NumCovariates();
    feature_presort_.resize(num_features_);
    for (int i = 0; i < num_features_; i++) {
      feature_presort_[i].reset(new FeaturePresortRoot(dataset, i, dataset->GetFeatureType(i)));
    }
  }

  ~FeaturePresortRootContainer() {}

  FeaturePresortRoot* GetFeaturePresort(int feature_num) {return feature_presort_[feature_num].get(); }

 private:
  std::vector<std::unique_ptr<FeaturePresortRoot>> feature_presort_;
  int num_features_;
};

/*! \brief Data structure that tracks pre-sorted feature values 
 *  through a tree's split lifecycle
 * 
 *  This class is initialized from a FeaturePresortRoot which has computed the 
 *  sort indices for a given feature over the entire dataset, so that sorting 
 *  is not necessary for each new tree.
 *  
 *  When a split is made, this class handles sifting for each feature, so that 
 *  the presorted feature values available at each node are easily queried.
 */
class FeaturePresortPartition {
 public:
  FeaturePresortPartition(FeaturePresortRoot* feature_presort_root, Dataset* dataset, int32_t feature_index, FeatureType feature_type) {
    // Unpack all feature details
    feature_index_ = feature_index;
    feature_type_ = feature_type;
    num_obs_ = dataset->NumObservations();
    feature_sort_indices_ = feature_presort_root->feature_sort_indices_;

    // Initialize new tree to root
    data_size_t node_offset = 0;
    node_offset_sizes_.emplace_back(node_offset, num_obs_);
  }

  ~FeaturePresortPartition() {}

  /*! \brief Split numeric / ordered categorical feature and update sort indices */
  void SplitFeatureNumeric(Dataset* dataset, int32_t node_id, int32_t feature_index, double split_value);

  /*! \brief Split unordered categorical feature and update sort indices */
  void SplitFeatureCategorical(Dataset* dataset, int32_t node_id, int32_t feature_index, std::vector<std::uint32_t> const& category_list);

  /*! \brief Start position of node indexed by node_id */
  data_size_t NodeBegin(int32_t node_id) {return node_offset_sizes_[node_id].Begin();}

  /*! \brief End position of node indexed by node_id */
  data_size_t NodeEnd(int32_t node_id) {return node_offset_sizes_[node_id].End();}

  /*! \brief Size (in observations) of node indexed by node_id */
  data_size_t NodeSize(int32_t node_id) {return node_offset_sizes_[node_id].Size();}

  /*! \brief Data indices for a given node */
  std::vector<data_size_t> NodeIndices(int node_id);

  /*! \brief Feature sort index j */
  data_size_t SortIndex(data_size_t j) {return feature_sort_indices_[j];}

  /*! \brief Feature type */
  FeatureType GetFeatureType() {return feature_type_;}

  /*! \brief Update SampleNodeMapper for all the observations in node_id */
  void UpdateObservationMapping(int node_id, int tree_id, SampleNodeMapper* sample_node_mapper);

 private:
  /*! \brief Add left and right nodes */
  void AddLeftRightNodes(data_size_t left_node_begin, data_size_t left_node_size, data_size_t right_node_begin, data_size_t right_node_size);

  /*! \brief Feature sort indices */
  std::vector<data_size_t> feature_sort_indices_;
  std::vector<NodeOffsetSize> node_offset_sizes_;
  int32_t feature_index_;
  FeatureType feature_type_;
  data_size_t num_obs_;
};

/*! \brief Data structure for tracking observations through a tree partition with each feature pre-sorted */
class SortedNodeSampleTracker {
 public:
  SortedNodeSampleTracker(FeaturePresortRootContainer* feature_presort_root_container, Dataset* dataset) {
    num_features_ = dataset->NumCovariates();
    feature_partitions_.resize(num_features_);
    FeaturePresortRoot* feature_presort_root;
    for (int i = 0; i < num_features_; i++) {
      feature_presort_root = feature_presort_root_container->GetFeaturePresort(i);
      feature_partitions_[i].reset(new FeaturePresortPartition(feature_presort_root, dataset, i, dataset->GetFeatureType(i)));
    }
  }

  /*! \brief Partition a node based on a new split rule */
  void PartitionNode(Dataset* dataset, int node_id, int feature_split, double split_value) {
    for (int i = 0; i < num_features_; i++) {
      feature_partitions_[i]->SplitFeatureNumeric(dataset, node_id, feature_split, split_value);
    }
  }

  /*! \brief Partition a node based on a new split rule */
  void PartitionNode(Dataset* dataset, int node_id, int feature_split, std::vector<std::uint32_t> const& category_list) {
    for (int i = 0; i < num_features_; i++) {
      feature_partitions_[i]->SplitFeatureCategorical(dataset, node_id, feature_split, category_list);
    }
  }

  /*! \brief First index of data points contained in node_id */
  data_size_t NodeBegin(int node_id, int feature_index) {
    return feature_partitions_[feature_index]->NodeBegin(node_id);
  }

  /*! \brief One past the last index of data points contained in node_id */
  data_size_t NodeEnd(int node_id, int feature_index) {
    return feature_partitions_[feature_index]->NodeEnd(node_id);
  }

  /*! \brief Data indices for a given node */
  std::vector<data_size_t> NodeIndices(int node_id, int feature_index) {
    return feature_partitions_[feature_index]->NodeIndices(node_id);
  }

  /*! \brief Feature sort index j for feature_index */
  data_size_t SortIndex(data_size_t j, int feature_index) {return feature_partitions_[feature_index]->SortIndex(j); }

  /*! \brief Update SampleNodeMapper for all the observations in node_id */
  void UpdateObservationMapping(int node_id, int tree_id, SampleNodeMapper* sample_node_mapper, int feature_index = 0) {
    feature_partitions_[feature_index]->UpdateObservationMapping(node_id, tree_id, sample_node_mapper);
  }

 private:
  std::vector<std::unique_ptr<FeaturePresortPartition>> feature_partitions_;
  int num_features_;
};

} // namespace StochTree

#endif // STOCHTREE_NODE_SAMPLE_TRACKER_H_