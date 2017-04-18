
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

# H2O Gradient Boosting Machine with default settings
model <- h2o.gbm(x = features, 
                 y = target, 
                 training_frame = h_train,
                 seed = 1234)

# Print out model summary
summary(model)

# Check performance on test set
h2o.performance(model, h_test)

# Use the model for predictions
yhat_test <- h2o.predict(model, h_test)

# Show first 10 rows
head(yhat_test, 10)
