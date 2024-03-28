/*! Copyright (c) 2024 stochtree authors*/
#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/json11.h>
#include <stochtree/leaf_model.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>

#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace StochTree{

enum ForestLeafModel {
    kConstant, 
    kUnivariateRegression, 
    kMultivariateRegression
};

void GenerateRandomData(std::vector<double>& covariates, std::vector<double>& basis, std::vector<double>& outcome, std::vector<double>& rfx_basis, std::vector<int32_t>& rfx_groups, int n, int x_cols, int omega_cols, int y_cols, int rfx_basis_cols) {
  std::mt19937 gen(101);
  std::uniform_real_distribution<double> uniform_dist{0.0,1.0};
  std::normal_distribution<double> normal_dist(0.,1.);
  std::vector<double> betas{-10, -5, 5, 10};
  int num_partitions = betas.size();
  double beta = 5;
  double f_x_omega;
  double rfx;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < x_cols; j++) {
      covariates[i*x_cols + j] = uniform_dist(gen);
    }
    
    for (int j = 0; j < omega_cols; j++) {
      basis[i*omega_cols + j] = uniform_dist(gen);
    }
    
    for (int j = 0; j < rfx_basis_cols; j++) {
      rfx_basis[i*rfx_basis_cols + j] = 1;
    }
    
    if (i % 2 == 0) {
      rfx_groups[i] = 1;
    } else {
      rfx_groups[i] = 2;
    }
    
    for (int j = 0; j < y_cols; j++) {
      if ((covariates[i * x_cols + 0] >= 0.0) && covariates[i * x_cols + 0] < 0.25) {
        f_x_omega = betas[0] * basis[i * omega_cols + 0];
      } else if ((covariates[i * x_cols + 0] >= 0.25) && covariates[i * x_cols + 0] < 0.5) {
        f_x_omega = betas[1] * basis[i * omega_cols + 0];
      } else if ((covariates[i * x_cols + 0] >= 0.5) && covariates[i * x_cols + 0] < 0.75) {
        f_x_omega = betas[2] * basis[i * omega_cols + 0];
      } else {
        f_x_omega = betas[3] * basis[i * omega_cols + 0];
      }
      if (rfx_groups[i] == 1) {
        rfx = 5.;
      } else {
        rfx = -5.;
      }
      outcome[i * y_cols + j] = f_x_omega + rfx + 0.1*normal_dist(gen);
    }
  }
}

void OutcomeOffsetScale(ColumnVector& residual, double& outcome_offset, double& outcome_scale) {
  data_size_t n = residual.NumRows();
  double outcome_val = 0.0;
  double outcome_sum = 0.0;
  double outcome_sum_squares = 0.0;
  double var_y = 0.0;
  for (data_size_t i = 0; i < n; i++){
    outcome_val = residual.GetElement(i);
    outcome_sum += outcome_val;
    outcome_sum_squares += std::pow(outcome_val, 2.0);
  }
  var_y = outcome_sum_squares / static_cast<double>(n) - std::pow(outcome_sum / static_cast<double>(n), 2.0);
  outcome_scale = std::sqrt(var_y);
  outcome_offset = outcome_sum / static_cast<double>(n);
  double previous_residual;
  for (data_size_t i = 0; i < n; i++){
    previous_residual = residual.GetElement(i);
    residual.SetElement(i, (previous_residual - outcome_offset) / outcome_scale);
  }
}

void sampleGFR(ForestTracker& tracker, TreePrior& tree_prior, ForestContainer& forest_samples, ForestDataset& dataset, 
               ColumnVector& residual, std::mt19937& rng, std::vector<FeatureType>& feature_types, std::vector<double>& var_weights_vector, 
               ForestLeafModel leaf_model_type, Eigen::MatrixXd& leaf_scale_matrix, double global_variance, double leaf_scale, int cutpoint_grid_size) {
  if (leaf_model_type == ForestLeafModel::kConstant) {
    GaussianConstantLeafModel leaf_model = GaussianConstantLeafModel(leaf_scale);
    GFRForestSampler<GaussianConstantLeafModel> sampler = GFRForestSampler<GaussianConstantLeafModel>(cutpoint_grid_size);
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance, feature_types);
  } else if (leaf_model_type == ForestLeafModel::kUnivariateRegression) {
    GaussianUnivariateRegressionLeafModel leaf_model = GaussianUnivariateRegressionLeafModel(leaf_scale);
    GFRForestSampler<GaussianUnivariateRegressionLeafModel> sampler = GFRForestSampler<GaussianUnivariateRegressionLeafModel>(cutpoint_grid_size);
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance, feature_types);
  } else if (leaf_model_type == ForestLeafModel::kMultivariateRegression) {
    GaussianMultivariateRegressionLeafModel leaf_model = GaussianMultivariateRegressionLeafModel(leaf_scale_matrix);
    GFRForestSampler<GaussianMultivariateRegressionLeafModel> sampler = GFRForestSampler<GaussianMultivariateRegressionLeafModel>(cutpoint_grid_size);
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance, feature_types);
  }
}

void sampleMCMC(ForestTracker& tracker, TreePrior& tree_prior, ForestContainer& forest_samples, ForestDataset& dataset, 
                ColumnVector& residual, std::mt19937& rng, std::vector<FeatureType>& feature_types, std::vector<double>& var_weights_vector, 
                ForestLeafModel leaf_model_type, Eigen::MatrixXd& leaf_scale_matrix, double global_variance, double leaf_scale, int cutpoint_grid_size) {
  if (leaf_model_type == ForestLeafModel::kConstant) {
    GaussianConstantLeafModel leaf_model = GaussianConstantLeafModel(leaf_scale);
    MCMCForestSampler<GaussianConstantLeafModel> sampler = MCMCForestSampler<GaussianConstantLeafModel>();
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance);
  } else if (leaf_model_type == ForestLeafModel::kUnivariateRegression) {
    GaussianUnivariateRegressionLeafModel leaf_model = GaussianUnivariateRegressionLeafModel(leaf_scale);
    MCMCForestSampler<GaussianUnivariateRegressionLeafModel> sampler = MCMCForestSampler<GaussianUnivariateRegressionLeafModel>();
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance);
  } else if (leaf_model_type == ForestLeafModel::kMultivariateRegression) {
    GaussianMultivariateRegressionLeafModel leaf_model = GaussianMultivariateRegressionLeafModel(leaf_scale_matrix);
    MCMCForestSampler<GaussianMultivariateRegressionLeafModel> sampler = MCMCForestSampler<GaussianMultivariateRegressionLeafModel>();
    sampler.SampleOneIter(tracker, forest_samples, leaf_model, dataset, residual, tree_prior, rng, var_weights_vector, global_variance);
  }
}

void RunAPI() {
  // Data dimensions
  int n = 1000;
  int x_rows = n;
  int x_cols = 2;
  int omega_rows = n;
  int omega_cols = 1;
  int y_rows = n;
  int y_cols = 1;
  int rfx_basis_rows = n;
  int rfx_basis_cols = 1;
  int num_rfx_groups = 2;
  
  // Declare covariates, basis and outcome
  std::vector<double> covariates_raw(x_rows*x_cols);
  std::vector<double> basis_raw(omega_rows*omega_cols);
  std::vector<double> rfx_basis_raw(rfx_basis_rows*rfx_basis_cols);
  std::vector<double> outcome_raw(y_rows*y_cols);
  std::vector<int32_t> rfx_groups(n);
  
  // Load the data
  GenerateRandomData(covariates_raw, basis_raw, outcome_raw, rfx_basis_raw, rfx_groups, n, x_cols, omega_cols, y_cols, rfx_basis_cols);
  
  // Define internal datasets
  bool row_major = true;

  // Construct datasets for training
  ForestDataset dataset = ForestDataset();
  dataset.AddCovariates(covariates_raw.data(), n, x_cols, row_major);
  dataset.AddBasis(basis_raw.data(), n, omega_cols, row_major);
  ColumnVector residual = ColumnVector(outcome_raw.data(), n);
  
  // Center and scale the data
  double outcome_offset;
  double outcome_scale;
  OutcomeOffsetScale(residual, outcome_offset, outcome_scale);

  // // Construct a random effects dataset
  // RandomEffectsDataset rfx_dataset = RandomEffectsDataset();
  // rfx_dataset.AddBasis(rfx_basis_raw.data(), n, rfx_basis_cols, row_major);
  // rfx_dataset.AddGroupLabels(rfx_groups);
  
  // Initialize an ensemble
  int num_trees = 100;
  int output_dimension = 1;
  bool is_leaf_constant = false;
  ForestContainer forest_samples = ForestContainer(num_trees, output_dimension, is_leaf_constant);

  // // Initialize a leaf model
  // double leaf_prior_mean = 0.;
  // double leaf_prior_scale = 1.;
  // GaussianUnivariateRegressionLeafModel leaf_model = GaussianUnivariateRegressionLeafModel(leaf_prior_scale);

  // // Check consistency
  // CHECK(ForestModelCompatible(forest_samples, leaf_model));

  // // Initialize a global variance model
  // double a_global_variance = 1.;
  // double b_global_variance = 1.;
  // HomoskedasticVarianceModel global_variance_model = HomoskedasticVarianceModel(a_global_variance, b_global_variance);
  
  // Check consistency

  // Initialize forest sampling machinery
  std::vector<FeatureType> feature_types(x_cols, FeatureType::kNumeric);
  double alpha = 0.99;
  double beta = 1.25;
  int min_samples_leaf = 10;
  int cutpoint_grid_size = 100;
  double a_rfx = 1.;
  double b_rfx = 1.;
  double a_leaf = 2.;
  double b_leaf = 0.5;
  double nu = 4.;
  double lamb = 0.5;
  ForestLeafModel leaf_model_type = ForestLeafModel::kUnivariateRegression;

  // Set leaf model parameters
  double leaf_scale_init = 1.;
  Eigen::MatrixXd leaf_scale_matrix, leaf_scale_matrix_init;
  // leaf_scale_matrix_init << 1.0, 0.0, 0.0, 1.0;
  double leaf_scale;

  // Set global variance
  double global_variance_init = 1.0;
  double global_variance;

  // Set variable weights
  double const_var_wt = static_cast<double>(1/x_cols);
  std::vector<double> variable_weights(x_cols, const_var_wt);

  // Initialize tracker and tree prior
  ForestTracker tracker = ForestTracker(dataset.GetCovariates(), feature_types, num_trees, n);
  TreePrior tree_prior = TreePrior(alpha, beta, min_samples_leaf);

  // Initialize a random number generator
  int random_seed = 1234;
  std::mt19937 rng = std::mt19937(random_seed);
  
  // Initialize variance models
  GlobalHomoskedasticVarianceModel global_var_model = GlobalHomoskedasticVarianceModel();
  LeafNodeHomoskedasticVarianceModel leaf_var_model = LeafNodeHomoskedasticVarianceModel();

  // Initialize storage for samples of variance
  std::vector<double> global_variance_samples{};
  std::vector<double> leaf_variance_samples{};

  // Run the GFR sampler
  int num_gfr_samples = 10;
  for (int i = 0; i < num_gfr_samples; i++) {
    if (i == 0) {
      global_variance = global_variance_init;
      leaf_scale = leaf_scale_init;
    } else {
      global_variance = global_variance_samples[i-1];
      leaf_scale = leaf_variance_samples[i-1];
    }

    // Sample tree ensemble
    sampleGFR(tracker, tree_prior, forest_samples, dataset, residual, rng, feature_types, variable_weights, 
              leaf_model_type, leaf_scale_matrix, global_variance, leaf_scale, cutpoint_grid_size);

    // Sample leaf node variance
    leaf_variance_samples.push_back(leaf_var_model.SampleVarianceParameter(forest_samples.GetEnsemble(i), a_leaf, b_leaf, rng));

    // Sample global variance
    global_variance_samples.push_back(global_var_model.SampleVarianceParameter(residual.GetData(), nu, nu*lamb, rng));
  }

  // Run the MCMC sampler
  int num_mcmc_samples = 10;
  for (int i = num_gfr_samples; i < num_gfr_samples + num_mcmc_samples; i++) {
    if (i == 0) {
      global_variance = global_variance_init;
      leaf_scale = leaf_scale_init;
    } else {
      global_variance = global_variance_samples[i-1];
      leaf_scale = leaf_variance_samples[i-1];
    }

    // Sample tree ensemble
    sampleMCMC(tracker, tree_prior, forest_samples, dataset, residual, rng, feature_types, variable_weights, 
               leaf_model_type, leaf_scale_matrix, global_variance, leaf_scale, cutpoint_grid_size);

    // Sample leaf node variance
    leaf_variance_samples.push_back(leaf_var_model.SampleVarianceParameter(forest_samples.GetEnsemble(i), a_leaf, b_leaf, rng));

    // Sample global variance
    global_variance_samples.push_back(global_var_model.SampleVarianceParameter(residual.GetData(), nu, nu*lamb, rng));
  }

  // Quick check: tree json round trip
  int sample_num = 2;
  int tree_num = 3;
  Tree* tree = forest_samples.GetEnsemble(sample_num)->GetTree(tree_num);
  json11::Json tree_json = tree->to_json();
  std::cout << tree_json.dump() << std::endl;
  Tree new_tree = Tree();
  new_tree.from_json(tree_json);
  json11::Json new_tree_json = new_tree.to_json();
  std::cout << new_tree_json.dump() << std::endl;
}

} // namespace StochTree

int main() {
  StochTree::RunAPI();
}
