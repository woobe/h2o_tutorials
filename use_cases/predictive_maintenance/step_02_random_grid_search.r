
# Load h2o library
suppressPackageStartupMessages(library(h2o))

# Start and connect to a local H2O cluster
h2o.init(nthreads = -1)

# Importing data from local CSV
h_secom <- h2o.importFile(path = "secom.csv", destination_frame = "h_secom")

# Print out column names
colnames(h_secom)

# Look at "Classification"
summary(h_secom$Classification, exact_quantiles=TRUE)

# "Classification" is a column of numerical values
# Convert "Classification" in secom dataset from numerical to categorical value
h_secom$Classification <- as.factor(h_secom$Classification)

# Look at "Classification" again
summary(h_secom$Classification, exact_quantiles=TRUE)

# Define target (y) and features (x)
target <- "Classification"
features <- setdiff(colnames(h_secom), target)
print(features)

# Splitting dataset into training and test
h_split <- h2o.splitFrame(h_secom, ratios = 0.7, seed = 1234)
h_train <- h_split[[1]] # 70%
h_test  <- h_split[[2]] # 30%

# Look at the size
dim(h_train)
dim(h_test)

# Check Classification in each dataset
summary(h_train$Classification, exact_quantiles = TRUE)
summary(h_test$Classification, exact_quantiles = TRUE)

# Define the criteria for random grid search
search_criteria = list(strategy = "RandomDiscrete",
                       max_models = 10,   
                       seed = 1234)

# Define the range of hyper-parameters for grid search
hyper_params <- list(
    sample_rate = c(0.6, 0.7, 0.8, 0.9),
    col_sample_rate = c(0.6, 0.7, 0.8, 0.9),
    max_depth = c(4, 5, 6)
)

# Set up grid search
# Add a seed for reproducibility
rand_grid <- h2o.grid(
  
    # Core parameters for model training
    x = features,
    y = target,
    training_frame = h_train,
    ntrees = 500,
    learn_rate = 0.05,
    balance_classes = TRUE,
    seed = 1234,
    
    # Settings for Cross-Validation
    nfolds = 5,
    fold_assignment = "Stratified",
    
    # Parameters for early stopping
    stopping_metric = "mean_per_class_error",
    stopping_rounds = 15,
    score_tree_interval = 1,
        
    # Parameters for grid search
    grid_id = "rand_grid",
    hyper_params = hyper_params,
    algorithm = "gbm",
    search_criteria = search_criteria  
  
)

# Sort and show the grid search results
rand_grid <- h2o.getGrid(grid_id = "rand_grid", sort_by = "mean_per_class_error", decreasing = FALSE)
print(rand_grid)

# Extract the best model from random grid search
best_model_id <- rand_grid@model_ids[[1]] # top of the list
best_model <- h2o.getModel(best_model_id)
print(best_model)

# Check performance on test set
h2o.performance(best_model, h_test)

# Use the model for predictions
yhat_test <- h2o.predict(best_model, h_test)

# Show first 10 rows
head(yhat_test, 10)
