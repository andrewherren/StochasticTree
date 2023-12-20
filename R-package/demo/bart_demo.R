################################################################################
## Brief BART demo script
################################################################################

# Load stochtree
library(stoch.tree)

# Set seed
random_seed = 12
set.seed(random_seed)

# Run a single iteration of the comparison and plot the results

# Generate simulated data and split into training and prediction sets
n <- 500
p <- 10
num_samples <- 50
num_burnin <- 100
num_trees <- 100
train_inds <- sample(1:n, round(n*0.8), replace = F)
test_inds <- (1:n)[!((1:n) %in% train_inds)]
X <- matrix(runif(n*p), ncol=p)
y <- X[,1] * 5 + X[,2] * 100 + rnorm(n, 0, 1)
Xtrain <- X[train_inds,]
Xtest <- X[test_inds,]
ytrain <- y[train_inds]
ytest <- y[test_inds]
model_matrix_train <- cbind(ytrain, Xtrain)
model_matrix_test <- cbind(ytest, Xtest)

# Sample from BART and evaluate its predictions on out of sample data
param_list = list(outcome_columns=0, num_trees=num_trees, num_burnin=num_burnin, num_samples=num_samples, min_data_in_leaf=1, alpha=0.95, beta=1.25, random_seed=random_seed)
stochtree_samples <- bart(model_matrix_train, param_list)
stochtree_predictions <- predict(stochtree_samples, model_matrix_test, param_list)
stochtree_avg_prediction <- rowMeans(stochtree_predictions)
stochtree_rmse <- sqrt(mean((stochtree_avg_prediction - ytest)^2))
for (j in 1:ncol(stochtree_predictions)) {plot(ytest, stochtree_predictions[,j], ylab = paste0("yhat_",j)); abline(0,1); Sys.sleep(0.25)}
plot(ytest, rowMeans(stochtree_predictions), ylab = "yhat_mean"); abline(0,1)
