/*!
 * Derived from xgboost tree unit test code:
 * https://github.com/dmlc/xgboost/blob/master/tests/cpp/tree/test_tree_model.cc
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <stochtree/log.h>
#include <stochtree/tree.h>
#include <iostream>
#include <memory>

TEST(Tree, UnivariateTreeConstruction) {
  StochTree::Tree tree;
  tree.Init(1);
  ASSERT_EQ(tree.LeafValue(0), 0.);
  tree.ExpandNode(0, 0, 0., true, 0., 0.);
  ASSERT_EQ(tree.NumNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kNumericalSplitNode);
  tree.CollapseToLeaf(0, 0.);
  ASSERT_EQ(tree.NumValidNodes(), 1);
  tree.ExpandNode(0, 0, 0., true, 0., 0.);
  ASSERT_EQ(tree.NumValidNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kNumericalSplitNode);
  ASSERT_EQ(tree.NumLeaves(), 2);
  ASSERT_FALSE(tree.IsLeaf(0));
  ASSERT_TRUE(tree.IsLeaf(1));
  ASSERT_TRUE(tree.IsLeaf(2));
}

TEST(Tree, BadInitialization) {
  StochTree::Tree tree;
  EXPECT_THROW(tree.Init(0), std::runtime_error);
  EXPECT_THROW(tree.Init(-1), std::runtime_error);
}

TEST(Tree, UnivariateTreeCategoricalSplitConstruction) {
  StochTree::Tree tree;
  tree.Init(1);
  ASSERT_EQ(tree.LeafValue(0), 0.);
  tree.ExpandNode(0, 0, std::vector<std::uint32_t>{1,4,6}, true, 0., 0.);
  ASSERT_EQ(tree.NumNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kCategoricalSplitNode);
  tree.CollapseToLeaf(0, 0.);
  ASSERT_EQ(tree.NumValidNodes(), 1);
  tree.ExpandNode(0, 0, std::vector<std::uint32_t>{2,3,5}, true, 0., 0.);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kCategoricalSplitNode);
  ASSERT_EQ(tree.NumValidNodes(), 3);
  ASSERT_EQ(tree.NumLeaves(), 2);
  ASSERT_FALSE(tree.IsLeaf(0));
  ASSERT_TRUE(tree.IsLeaf(1));
  ASSERT_TRUE(tree.IsLeaf(2));
}

TEST(Tree, MultivariateTreeConstruction) {
  StochTree::Tree tree;
  int tree_dim = 2;
  std::vector<double> leaf_values(tree_dim, 0.);
  tree.Init(tree_dim);
  EXPECT_THROW(tree.ExpandNode(0, 0, 0., true, 0., 0.), std::runtime_error);
  ASSERT_EQ(tree.LeafVector(0), leaf_values);
  tree.ExpandNode(0, 0, 0., true, leaf_values, leaf_values);
  ASSERT_EQ(tree.NumNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kNumericalSplitNode);
  EXPECT_THROW(tree.CollapseToLeaf(0, 0.);, std::runtime_error);
  tree.CollapseToLeaf(0, leaf_values);
  ASSERT_EQ(tree.NumValidNodes(), 1);
  tree.ExpandNode(0, 0, 0., true, leaf_values, leaf_values);
  ASSERT_EQ(tree.NumValidNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kNumericalSplitNode);
  ASSERT_EQ(tree.NumLeaves(), 2);
  ASSERT_FALSE(tree.IsLeaf(0));
  ASSERT_TRUE(tree.IsLeaf(1));
  ASSERT_TRUE(tree.IsLeaf(2));
}

TEST(Tree, MultivariateTreeCategoricalSplitConstruction) {
  StochTree::Tree tree;
  int tree_dim = 2;
  std::vector<double> leaf_values(tree_dim, 0.);
  tree.Init(tree_dim);
  ASSERT_EQ(tree.LeafVector(0), leaf_values);
  tree.ExpandNode(0, 0, std::vector<std::uint32_t>{1,4,6}, true, leaf_values, leaf_values);
  ASSERT_EQ(tree.NumNodes(), 3);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kCategoricalSplitNode);
  tree.CollapseToLeaf(0, leaf_values);
  ASSERT_EQ(tree.NumValidNodes(), 1);
  tree.ExpandNode(0, 0, std::vector<std::uint32_t>{2,3,5}, true, leaf_values, leaf_values);
  ASSERT_EQ(tree.NodeType(0), StochTree::TreeNodeType::kCategoricalSplitNode);
  ASSERT_EQ(tree.NumValidNodes(), 3);
  ASSERT_EQ(tree.NumLeaves(), 2);
  ASSERT_FALSE(tree.IsLeaf(0));
  ASSERT_TRUE(tree.IsLeaf(1));
  ASSERT_TRUE(tree.IsLeaf(2));
}
