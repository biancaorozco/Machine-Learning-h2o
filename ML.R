# MACHINE LEARNING ----

# Objectives:
#   Size the problem
#   Prepare the data for Binary Classification
#   Build models with H2O: GLM, GBM, RF
#   Inspect Features with LIME

# Estimated time: 2-3 hours



# 1.0 LIBRARIES ----
library(tidyverse)   # Workhorse with dplyr, ggplot2, etc
library(h2o)         # High Performance Machine Learning
library(recipes)     # Preprocessing
library(rsample)     # Sampling
install.packages('lime')
library(lime)        # Black-box explanations


# 2.0 DATA ----

unzip("00_data/application_train.csv.zip", exdir = "00_data/")
unzip("00_data/HomeCredit_columns_description.csv.zip", exdir = "00_data/")

# Loan Applications
application_train_raw_tbl <- read_csv("00_data/application_train.csv")

application_train_raw_tbl
#This created a tibble (dataset/spreadsheet in the data console -->)

glimpse(application_train_raw_tbl) 
#Gives you a glimpse of your raw data so you can check for missing data
#This allows you to see your 'features' or variables 
#Next step would usually be to run a coorelation analysis to see which variables are most important

# Column (Feature) Descriptions
feature_description_tbl <- read_csv("00_data/HomeCredit_columns_description.csv")

feature_description_tbl

# 3.0 SIZE THE PROBLEM ----

count(application_train_raw_tbl, TARGET) #This is the same as lines 47-48, but doesn't use pipe

# How many defaulters?
application_train_raw_tbl %>% #<-- are called pipe, it is for human readability
    count(TARGET) %>%
    #mutate(n_total = n / 0.15) %>%
    mutate(pct = n / sum(n)) %>% #run lines 47-50 and this shows you the relative frequency by mutating (adding a column of) the count
    mutate(pct_text = scales::percent(pct)) #this line changes the relative frequency into a percentage

#Note: Companies don't care about count/frequency, proportion/relative frequency, or percentages
#Companies want $$$$$$$$$$$$$$$

# Size the problem financially $$$
size_problem_tbl <- application_train_raw_tbl %>%
    count(TARGET) %>%
    filter(TARGET == 1) %>%
    # approximate number of annual defaults
    #mutate(prop = 0.15,
          # n_total = n / prop) %>%
    # cost of default
    mutate(avg_loan = 15000,
           avg_recovery = 0.40 * avg_loan,
           avg_loss = avg_loan - avg_recovery) %>%
    mutate(total_loss = n * avg_loss) %>%
    mutate(total_loss_text = scales::dollar(total_loss))

size_problem_tbl
#Note: Total annual $ lost is $223,425,000
#If you saved the company 10%, that would be $22,342,500 which is exactly what DS can do for companies


# 4.0 EXPLORATORY DATA ANALYSIS (SKIPPED) ----
#   SKIPPED - Very Important!
#   Efficient exploration of features to find which to focus on
#   Critical Step in Business Science Problem Framework
#   Taught in my DS4B 201-R Course
#   IMPORTANT: ATTEND MY TALK TOMORROW


# 5.0 SPLIT DATA ----

# Resource: https://tidymodels.github.io/rsample/

set.seed(1234) #random seed: if you're working with big data, it's better to sample it to save time and memory 
split_obj_1 <- initial_split(application_train_raw_tbl, 
                             strata = "TARGET", #makes random sample have the same proportion as our big data (91.9% and 8.1%)
                             prop = 0.2) #splits data sample into 20% / 80%
#Train/Test Split (Take 80% of the 20%)
set.seed(1234) 
split_obj_2 <- initial_split(training(split_obj_1), 
                             strata = "TARGET", 
                             prop = 0.8)

# Working with 20% sample of "Big Data"
train_raw_tbl <- training(split_obj_2) # 80% of Data
test_raw_tbl  <- testing(split_obj_2)  # 20% of Data

# Verify proportions have been maintained
train_raw_tbl %>%
    count(TARGET) %>%
    mutate(prop = n / sum(n))

test_raw_tbl %>%
    count(TARGET) %>%
    mutate(prop = n / sum(n))



# 6.0 PREPROCESSING ----

# Fix issues with data: 
#   Some Numeric data with low number of unique values should be Factor (Categorical)
#   All Character data should be Factor (Categorical)
#   NA's (imputation)

# 6.1 Handle Categorical ----

# Numeric
num2factor_names <- train_raw_tbl %>%
    select_if(is.numeric) %>% #Easy to understand, just read it.
    map_df(~ unique(.) %>% length()) %>% #Unique function counts unique observations wihtout counting repeats
    gather() %>% #transposes data
    arrange(value) %>% #lets you see what should be categorical (low number of unique observations)
    filter(value <= 6) %>%
    pull(key)

num2factor_names #These are the features (variables) that need to be categorical

# Character
string2factor_names <- train_raw_tbl %>%
    select_if(is.character) %>%
    names() #Gets the names of all the columns

string2factor_names


# 6.2 Missing Data ----

# Transform
missing_tbl <- train_raw_tbl %>%
    summarize_all(.funs = funs(sum(is.na(.)) / length(.))) %>% #This calculates the proportion of NAs/notNAs
    gather() %>%
    arrange(desc(value))

missing_tbl #Notice: 70% of the data is missing!!!! 

# Visualize Missing Data
missing_tbl %>%
    filter(value > 0) %>%
    mutate(key = as_factor(key) %>% fct_rev()) %>%
    ggplot(aes(x = value, y = key)) +
    geom_point() +
    geom_segment(aes(xend = 0, yend = key)) +
    expand_limits(x = c(0, 1)) +
    scale_x_continuous(labels = scales::percent) +
    labs(title = "Percentage Missing") 


# 6.3 Recipes ----

# Resource: https://tidymodels.github.io/recipes/

# recipe
rec_obj <- recipe(TARGET ~ ., data = train_raw_tbl) %>%
    step_num2factor(num2factor_names) %>% #Creates the plan to transform numerical factors that need to be categorical
    step_string2factor(string2factor_names) %>% #Creates the plan to transform the string factors that need to be categorical
    step_meanimpute(all_numeric()) %>% #This is going to replace the NAs with the avg of the  numerical observations (mean)
    step_modeimpute(all_nominal()) %>% #This is going to replace the NAs with the most often occuring categorical observation (the mode)
    prep(stringsAsFactors = FALSE)

# bake
train_tbl <- bake(rec_obj, train_raw_tbl) #This means to apply our plan or "recipe" to the dataset or "raw ingredients"
test_tbl  <- bake(rec_obj, test_raw_tbl) #Same

train_tbl %>% 
    glimpse()

# 7.0 MODELING -----

# 7.1 H2O Setup ----

# H2O Docs: http://docs.h2o.ai

h2o.init()

train_h2o <- as.h2o(train_tbl)
test_h2o  <- as.h2o(test_tbl)

y <- "TARGET"
x <- setdiff(names(train_h2o), y)

# 7.2 H2O Models ----

# 7.2.1 GLM (Elastic Net) ----

start <- Sys.time()
h2o_glm <- h2o.glm(
    x = x,
    y = y,
    training_frame   = train_h2o,
    validation_frame = test_h2o,
    nfolds = 5,
    seed   = 1234,
    
    # GLM
    family = "binomial"
    
)
Sys.time() - start
# Time difference of 6.508575 secs

h2o.performance(h2o_glm, valid = TRUE) %>%
    h2o.auc()
# [1] 0.7384649

h2o_glm@allparameters

# 7.2.2 GBM ----

# Resource: https://blog.h2o.ai/2016/06/h2o-gbm-tuning-tutorial-for-r/

start <- Sys.time()
h2o_gbm <- h2o.gbm( #This is a tree based model which is nonlinear
    x = x,
    y = y,
    training_frame   = train_h2o,
    validation_frame = test_h2o,
    nfolds = 5,
    seed   = 1234,
    
    # GBM
    ntrees     = 100,
    max_depth  = 5, #How many 'branch' layers this tree goes down
    learn_rate = 0.1 #The smaller the learning rate, the weight is changed 
)
Sys.time() - start
# Time difference of 29.29766 secs

h2o.performance(h2o_gbm, valid = TRUE) %>%
    h2o.auc()
# [1] 0.7369739

h2o_gbm@allparameters

# 7.2.3 Random Forest ----

start <- Sys.time()
h2o_rf <- h2o.randomForest(
    x = x,
    y = y,
    training_frame = train_h2o,
    validation_frame = test_h2o,
    nfolds = 5,
    seed = 1234,
    
    # RF
    ntrees          = 100,
    max_depth       = 5
    
)
Sys.time() - start
# Time difference of 27.21049 secs

h2o.performance(h2o_rf, valid = TRUE) %>%
    h2o.auc()
# [1] 0.7259596

h2o_rf@allparameters


# CHALLENGE: DEEP LEARNING ----

# 10 Minutes
# Create a Deep Learning Algorithm with H2O
# h2o.deeplearning
# 10 epochs
# 3 hidden layers: 100, 50, 10,



# 7.3 Saving & Loading Models ----

h2o.saveModel(h2o_gbm, "00_models")

h2o.loadModel("00_models/GBM_model_R_1539386726198_21")

# 8.0 Making Predictions -----

prediction_h2o <- h2o.predict(h2o_gbm, newdata = test_h2o) #Which loan applicants are high risk

prediction_tbl <- prediction_h2o %>%
    as.tibble() %>%
    bind_cols(
        test_tbl %>% select(TARGET, SK_ID_CURR)
    )

prediction_tbl

prediction_tbl %>%
    filter(TARGET == "1")


# 9.0 PERFORMANCE (SKIPPING) -----

#   Very Important
#   Adjusting Threshold
#   ROC Plot, Precision vs Recall
#   Gain & Lift - Important for executives

h2o_gbm %>%
    h2o.performance(valid = TRUE)


# 10.0 EXPLANATIONS LIME ----

# Create explainer
explainer <- train_tbl %>%
    select(-TARGET) %>%
    lime(
        model           = h2o_gbm,
        bin_continuous  = TRUE, #Always leave as continuous
        n_bins          = 4,
        quantile_bins   = TRUE
    )

# Create explanation
explanation <- test_tbl %>%
    filter(TARGET == "1") %>%
    slice(1) %>%
    select(-TARGET) %>%
    lime::explain(
        explainer = explainer,
        n_features = 8,
        n_permutations = 10000,
        dist_fun = "gower",
        kernel_width   = 1.5,
        feature_select = "lasso_path",
        # n_labels   = 2,
        labels         = "p1"
    )

explanation %>%
    as.tibble() %>%
    glimpse()

# Visualize
plot_features(explanation)


# What are Ext_Source?

feature_description_tbl %>%
    filter(str_detect(Row, "EXT_SOURCE")) %>%
    View()

# Equifax, Experian, TransUnion

# 11.0 OPTIMIZATION (SKIPPING) ----

#   Expected Value
#   Threshold Optimization - Find the balance of False Positives & False Negatives that maximizes revenue
#   Sensitivity Analysis - Taking into account what assumptions we are inputing into the model


# 12.0 RECOMMENDATION ALGORITHMS (SKIPPING) ----

#   3 Step Process:
#       1. Discretized Correlation Visualization (Correlation Funnel)
#       2. Fill out our Recommendation Algorithm Worksheet
#       3. Implement Strategies into R Code
#   Correlation Funnel - S&P Loved This!!

# BONUS #1: GRIDSEARCH ----

# GBM hyperparamters
gbm_params <- list(learn_rate       = c(0.01, 0.1),
                   max_depth        = c(3, 5, 9))
gbm_params

# Train and validate a cartesian grid of GBMs
gbm_grid <- h2o.grid("gbm", 
                     x = x, 
                     y = y,
                     grid_id = "gbm_grid1",
                     training_frame   = train_h2o,
                     validation_frame = test_h2o,
                     ntrees = 100,
                     seed   = 1234,
                     hyper_params = gbm_params)

h2o.getGrid(grid_id = "gbm_grid",
            sort_by = "auc",
            decreasing = TRUE)

h2o.getModel("gbm_grid1_model_1") %>%
    h2o.auc(valid = TRUE)
# [1] 0.7459666


# BONUS #2:  AUTOML ----

start <- Sys.time()
h2o_automl <- h2o.automl(
    x = x,
    y = y,
    training_frame = train_h2o,
    validation_frame = test_h2o,
    nfolds = 5,
    seed = 1234,
    
    # AutoML
    max_runtime_secs = 300
)
Sys.time() - start
# Time difference of 5.243099 mins


h2o_automl@leaderboard

h2o_automl@leader %>%
    h2o.auc(valid = TRUE)
# [1] 0.7423596
