/*!
 * Inspired by the design of the tree in the xgboost and treelite package, both released under the Apache license 
 * with the following copyright:
 * Copyright 2015-2023 by XGBoost Contributors
 * Copyright 2017-2021 by [treelite] Contributors
 */
#include <stochtree/common.h>
#include <stochtree/tree.h>

#include <algorithm>
#include <sstream>

namespace StochTree {

constexpr std::int32_t Tree::kInvalidNodeId;
constexpr std::int32_t Tree::kDeletedNodeMarker;
constexpr std::int32_t Tree::kRoot;

std::int32_t Tree::NumLeaves() const {
  std::int32_t leaves { 0 };
  auto const& self = *this;
  this->WalkTree([&leaves, &self](std::int32_t nidx) {
                   if (self.IsLeaf(nidx)) {
                     leaves++;
                   }
                   return true;
                 });
  return leaves;
}

std::int32_t Tree::NumLeafParents() const {
  std::int32_t leaf_parents { 0 };
  auto const& self = *this;
  this->WalkTree([&leaf_parents, &self](std::int32_t nidx) {
                   if (!self.IsLeaf(nidx)){
                     if ((self.IsLeaf(self.LeftChild(nidx))) && (self.IsLeaf(self.RightChild(nidx)))){
                      leaf_parents++;
                     }
                   }
                   return true;
                 });
  return leaf_parents;
}

std::int32_t Tree::NumSplitNodes() const {
  std::int32_t splits { 0 };
  auto const& self = *this;
  this->WalkTree([&splits, &self](std::int32_t nidx) {
                   if (!self.IsLeaf(nidx)){
                     splits++;
                   }
                   return true;
                 });
  return splits;
}

void Tree::InplacePredictFromNodes(std::vector<double> result, std::vector<std::int32_t> node_indices) {
  if (result.size() != node_indices.size()) {
    Log::Fatal("Indices and result vector are different sizes");
  }
  data_size_t n = node_indices.size();
  for (data_size_t i = 0; i < n; i++) {
    result[i] = this->LeafValue(node_indices[i]);
  }
}

double Tree::PredictFromNode(std::int32_t node_id) {
  if (!this->IsLeaf(node_id)) {
    Log::Fatal("Node %d is not a leaf node", node_id);
  }
  return this->LeafValue(node_id);
}

std::vector<double> Tree::PredictFromNodes(std::vector<std::int32_t> node_indices) {
  data_size_t n = node_indices.size();
  std::vector<double> result(n);
  for (data_size_t i = 0; i < n; i++) {
    result[i] = PredictFromNode(node_indices[i]);
  }
  return result;
}

Tree* Tree::Clone() {
  // Create tree with empty vectors / default scalar values
  Tree tree{};
  tree.Reset();

  // Copy vectors from existing tree
  tree.num_nodes = num_nodes;
  tree.num_deleted_nodes = num_deleted_nodes;

  tree.node_type_ = node_type_;
  tree.parent_ = parent_;
  tree.cleft_ = cleft_;
  tree.cright_ = cright_;
  tree.split_index_ = split_index_;
  tree.default_left_ = default_left_;
  tree.leaf_value_ = leaf_value_;
  tree.threshold_ = threshold_;
  tree.internal_nodes_ = internal_nodes_;
  tree.leaves_ = leaves_;
  tree.leaf_parents_ = leaf_parents_;
  tree.deleted_nodes_ = deleted_nodes_;

  tree.leaf_vector_ = leaf_vector_;
  tree.leaf_vector_begin_ = leaf_vector_begin_;
  tree.leaf_vector_end_ = leaf_vector_end_;
  tree.category_list_ = category_list_;
  tree.category_list_begin_ = category_list_begin_;
  tree.category_list_end_ = category_list_end_;

  tree.has_categorical_split_ = has_categorical_split_;
  tree.output_dimension_ = output_dimension_;

  return &tree;
}

void Tree::CloneFromTree(Tree* tree) {
  // Copy vectors from existing tree
  num_nodes = tree->num_nodes;
  num_deleted_nodes = tree->num_deleted_nodes;

  node_type_ = tree->node_type_;
  parent_ = tree->parent_;
  cleft_ = tree->cleft_;
  cright_ = tree->cright_;
  split_index_ = tree->split_index_;
  default_left_ = tree->default_left_;
  leaf_value_ = tree->leaf_value_;
  threshold_ = tree->threshold_;
  internal_nodes_ = tree->internal_nodes_;
  leaves_ = tree->leaves_;
  leaf_parents_ = tree->leaf_parents_;
  deleted_nodes_ = tree->deleted_nodes_;

  leaf_vector_ = tree->leaf_vector_;
  leaf_vector_begin_ = tree->leaf_vector_begin_;
  leaf_vector_end_ = tree->leaf_vector_end_;
  category_list_ = tree->category_list_;
  category_list_begin_ = tree->category_list_begin_;
  category_list_end_ = tree->category_list_end_;

  has_categorical_split_ = tree->has_categorical_split_;
  output_dimension_ = tree->output_dimension_;
}

std::int32_t Tree::AllocNode() {
  // Reuse a "deleted" node if available
  if (num_deleted_nodes != 0) {
    std::int32_t nid = deleted_nodes_.back();
    deleted_nodes_.pop_back();
    --num_deleted_nodes;
    return nid;
  }
  
  std::int32_t nd = num_nodes++;
  CHECK_LT(num_nodes, std::numeric_limits<int>::max());
  
  node_type_.push_back(TreeNodeType::kLeafNode);
  cleft_.push_back(kInvalidNodeId);
  cright_.push_back(kInvalidNodeId);
  split_index_.push_back(-1);
  default_left_.push_back(false);
  leaf_value_.push_back(static_cast<double>(0));
  threshold_.push_back(static_cast<double>(0));
  // THIS is a placeholder, currently set after AllocNode is called ... 
  // ... to be refactored ...
  parent_.push_back(static_cast<double>(0));

  leaf_vector_begin_.push_back(leaf_vector_.size());
  leaf_vector_end_.push_back(leaf_vector_.size());
  category_list_begin_.push_back(category_list_.size());
  category_list_end_.push_back(category_list_.size());

  return nd;
}

void Tree::DeleteNode(std::int32_t nid) {
  CHECK_GE(nid, 1);
  auto pid = this->Parent(nid);
  bool is_left = this->LeftChild(pid) == nid;
  if (is_left) {
    SetLeftChild(pid, kInvalidNodeId);
  } else {
    SetRightChild(pid, kInvalidNodeId);
  }

  deleted_nodes_.push_back(nid);
  ++num_deleted_nodes;

  // Remove from vectors that track leaves, leaf parents, internal nodes, etc...
  leaves_.erase(std::remove(leaves_.begin(), leaves_.end(), nid), leaves_.end());
  leaf_parents_.erase(std::remove(leaf_parents_.begin(), leaf_parents_.end(), nid), leaf_parents_.end());
  internal_nodes_.erase(std::remove(internal_nodes_.begin(), internal_nodes_.end(), nid), internal_nodes_.end());
}

void Tree::ExpandNode(std::int32_t nid, int split_index, double split_value, bool default_left, double left_value, double right_value) {
  CHECK_EQ(output_dimension_, 1);
  int pleft = this->AllocNode();
  int pright = this->AllocNode();
  this->SetChildren(nid, pleft, pright);
  this->SetParents(nid, pleft, pright);
  this->SetNumericSplit(nid, split_index, split_value, default_left);
  this->SetLeaf(pleft, left_value);
  this->SetLeaf(pright, right_value);

  // Remove nid from leaves and add to internal nodes and leaf parents
  leaves_.erase(std::remove(leaves_.begin(), leaves_.end(), nid), leaves_.end());
  leaf_parents_.push_back(nid);
  internal_nodes_.push_back(nid);

  // Remove nid's parent node (if applicable) from leaf parents
  if (!this->IsRoot(nid)){
    std::int32_t parent_idx = this->Parent(nid);
    leaf_parents_.erase(std::remove(leaf_parents_.begin(), leaf_parents_.end(), parent_idx), leaf_parents_.end());
  }

  // Add pleft and pright to leaves
  leaves_.push_back(pleft);
  leaves_.push_back(pright);
}

void Tree::ExpandNode(std::int32_t nid, int split_index, std::vector<std::uint32_t> const& categorical_indices, bool default_left, double left_value, double right_value) {
  CHECK_EQ(output_dimension_, 1);
  int pleft = this->AllocNode();
  int pright = this->AllocNode();
  this->SetChildren(nid, pleft, pright);
  this->SetParents(nid, pleft, pright);
  this->SetCategoricalSplit(nid, split_index, default_left, categorical_indices);
  this->SetLeaf(pleft, left_value);
  this->SetLeaf(pright, right_value);

  // Remove nid from leaves and add to internal nodes and leaf parents
  leaves_.erase(std::remove(leaves_.begin(), leaves_.end(), nid), leaves_.end());
  leaf_parents_.push_back(nid);
  internal_nodes_.push_back(nid);

  // Remove nid's parent node (if applicable) from leaf parents
  if (this->IsRoot(nid)){
    std::int32_t parent_idx = this->Parent(nid);
    leaf_parents_.erase(std::remove(leaf_parents_.begin(), leaf_parents_.end(), parent_idx), leaf_parents_.end());
  }

  // Add pleft and pright to leaves
  leaves_.push_back(pleft);
  leaves_.push_back(pright);
}

void Tree::ExpandNode(std::int32_t nid, int split_index, double split_value, bool default_left, std::vector<double> left_value_vector, std::vector<double> right_value_vector) {
  CHECK_GT(output_dimension_, 1);
  CHECK_EQ(output_dimension_, left_value_vector.size());
  CHECK_EQ(output_dimension_, right_value_vector.size());
  int pleft = this->AllocNode();
  int pright = this->AllocNode();
  this->SetChildren(nid, pleft, pright);
  this->SetParents(nid, pleft, pright);
  this->SetNumericSplit(nid, split_index, split_value, default_left);
  this->SetLeafVector(pleft, left_value_vector);
  this->SetLeafVector(pright, right_value_vector);

  // Remove nid from leaves and add to internal nodes and leaf parents
  leaves_.erase(std::remove(leaves_.begin(), leaves_.end(), nid), leaves_.end());
  leaf_parents_.push_back(nid);
  internal_nodes_.push_back(nid);

  // Remove nid's parent node (if applicable) from leaf parents
  if (!this->IsRoot(nid)){
    std::int32_t parent_idx = this->Parent(nid);
    leaf_parents_.erase(std::remove(leaf_parents_.begin(), leaf_parents_.end(), parent_idx), leaf_parents_.end());
  }

  // Add pleft and pright to leaves
  leaves_.push_back(pleft);
  leaves_.push_back(pright);
}

void Tree::ExpandNode(std::int32_t nid, int split_index, std::vector<std::uint32_t> const& categorical_indices, bool default_left, std::vector<double> left_value_vector, std::vector<double> right_value_vector) {
  CHECK_GT(output_dimension_, 1);
  CHECK_EQ(output_dimension_, left_value_vector.size());
  CHECK_EQ(output_dimension_, right_value_vector.size());
  int pleft = this->AllocNode();
  int pright = this->AllocNode();
  this->SetChildren(nid, pleft, pright);
  this->SetParents(nid, pleft, pright);
  this->SetCategoricalSplit(nid, split_index, default_left, categorical_indices);
  this->SetLeafVector(pleft, left_value_vector);
  this->SetLeafVector(pright, right_value_vector);

  // Remove nid from leaves and add to internal nodes and leaf parents
  leaves_.erase(std::remove(leaves_.begin(), leaves_.end(), nid), leaves_.end());
  leaf_parents_.push_back(nid);
  internal_nodes_.push_back(nid);

  // Remove nid's parent node (if applicable) from leaf parents
  if (this->IsRoot(nid)){
    std::int32_t parent_idx = this->Parent(nid);
    leaf_parents_.erase(std::remove(leaf_parents_.begin(), leaf_parents_.end(), parent_idx), leaf_parents_.end());
  }

  // Add pleft and pright to leaves
  leaves_.push_back(pleft);
  leaves_.push_back(pright);
}

void Tree::Reset() {
  // Clear all of the vectors that define the tree structure
  node_type_.clear();
  cleft_.clear();
  cright_.clear();
  split_index_.clear();
  default_left_.clear();
  leaf_value_.clear();
  threshold_.clear();
  parent_.clear();

  num_nodes = 0;
  has_categorical_split_ = false;

  leaf_vector_.clear();
  leaf_vector_begin_.clear();
  leaf_vector_end_.clear();
  category_list_.clear();
  category_list_begin_.clear();
  category_list_end_.clear();

  leaves_.clear();
  leaf_parents_.clear();
  internal_nodes_.clear();

  // Set bool / integer variables to default values
  num_nodes = 0;
  num_deleted_nodes = 0;
  has_categorical_split_ = false;
  output_dimension_ = 1;
}

void Tree::Init(std::int32_t output_dimension) {
  CHECK_GE(output_dimension, 1);

  // Clear all of the vectors that define the tree structure
  node_type_.clear();
  cleft_.clear();
  cright_.clear();
  split_index_.clear();
  default_left_.clear();
  leaf_value_.clear();
  threshold_.clear();
  parent_.clear();

  num_nodes = 0;
  has_categorical_split_ = false;

  leaf_vector_.clear();
  leaf_vector_begin_.clear();
  leaf_vector_end_.clear();
  category_list_.clear();
  category_list_begin_.clear();
  category_list_end_.clear();

  leaves_.clear();
  leaf_parents_.clear();
  internal_nodes_.clear();

  // Set output dimension
  output_dimension_ = output_dimension;

  // Allocate root node
  int rid = AllocNode();
  SetChildren(rid, kInvalidNodeId, kInvalidNodeId);
  SetParent(rid, kInvalidNodeId);
  if (output_dimension == 1) {
    this->SetLeaf(rid, 0.0);
  } else {
    this->SetLeafVector(rid, std::vector<double>(output_dimension, 0.));
  }

  // Add rid as a leaf node
  leaves_.push_back(rid);
}

void Tree::SetNumericSplit(std::int32_t nid, std::int32_t split_index, double threshold, bool default_left) {
  split_index_.at(nid) = split_index;
  threshold_.at(nid) = threshold;
  default_left_.at(nid) = default_left;
  node_type_.at(nid) = TreeNodeType::kNumericalSplitNode;
}

void Tree::SetCategoricalSplit(std::int32_t nid,
    std::int32_t split_index, bool default_left, std::vector<std::uint32_t> const& category_list) {
  // CHECK(CategoryList(nid).empty());
  std::size_t const begin = category_list_.size();
  std::size_t const end = begin + category_list.size();
  category_list_.insert(category_list_.end(), category_list.begin(), category_list.end());
  category_list_begin_.at(nid) = begin;
  category_list_end_.at(nid) = end;

  split_index_.at(nid) = split_index;
  default_left_.at(nid) = default_left;
  node_type_.at(nid) = TreeNodeType::kCategoricalSplitNode;

  has_categorical_split_ = true;
}

void Tree::SetLeaf(std::int32_t nid, double value) {
  CHECK_EQ(output_dimension_, 1);
  leaf_value_.at(nid) = value;
  cleft_.at(nid) = -1;
  cright_.at(nid) = -1;
  node_type_.at(nid) = TreeNodeType::kLeafNode;
}

void Tree::SetLeafVector(std::int32_t nid, std::vector<double> const& node_leaf_vector) {
  CHECK_GT(output_dimension_, 1);
  CHECK_EQ(output_dimension_, node_leaf_vector.size());
  if (HasLeafVector(nid)) {
    if (node_leaf_vector.size() != output_dimension_) {
      Log::Fatal("node_leaf_vector must be same size as the vector output dimension");
    }
    if (node_leaf_vector.size() != (leaf_vector_end_.at(nid) - leaf_vector_begin_.at(nid))) {
      Log::Fatal("Existing vector output is not the same size as node_leaf_vector");
    }
    std::size_t begin = leaf_vector_begin_.at(nid);
    std::size_t end = leaf_vector_end_.at(nid);
    std::size_t counter = 0;
    for (std::size_t i = begin; i < end; i++) {
      leaf_vector_[i] = node_leaf_vector[counter];
      counter++;
    }
  } else {
    std::size_t begin = leaf_vector_.size();
    std::size_t end = begin + node_leaf_vector.size();
    leaf_vector_.insert(leaf_vector_.end(), node_leaf_vector.begin(), node_leaf_vector.end());
    leaf_vector_begin_.at(nid) = begin;
    leaf_vector_end_.at(nid) = end;
  }

  split_index_.at(nid) = -1;
  cleft_.at(nid) = kInvalidNodeId;
  cright_.at(nid) = kInvalidNodeId;
  node_type_.at(nid) = TreeNodeType::kLeafNode;
}

} // namespace StochTree
