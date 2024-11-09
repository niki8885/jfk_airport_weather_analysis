# Load required libraries
library(yardstick)
library(tidymodels)
library(broom)
library(dplyr)
library(tidyverse)
library(rsample)
library(ggplot2)
library(glmnet)

# Load and inspect the data
jfk_weather_data <- read.csv("jfk_weather_sample.csv", header = TRUE, sep = ",")
glimpse(jfk_weather_data)

# Select relevant columns and rename for easier use
selected_df <- jfk_weather_data %>%
  select(HOURLYRelativeHumidity, HOURLYDRYBULBTEMPF, HOURLYPrecip, HOURLYWindSpeed, HOURLYStationPressure) %>%
  rename(relative_humidity = HOURLYRelativeHumidity,
         dry_bulb_temp_f = HOURLYDRYBULBTEMPF,
         precip = HOURLYPrecip,
         wind_speed = HOURLYWindSpeed,
         station_pressure = HOURLYStationPressure)

# Handle special values in precip (trace amounts and other symbols)
selected_df$precip <- ifelse(selected_df$precip == "T", 0.0, selected_df$precip)
selected_df$precip <- gsub("s$", "", selected_df$precip)
selected_df$precip <- as.numeric(selected_df$precip)

# Split the data into training and testing sets
set.seed(1234)
df_split <- initial_split(selected_df, prop = 0.8)
train_data <- training(df_split)
test_data <- testing(df_split)

# Filter out rows with missing values in training data
train_data_clean <- train_data %>%
  drop_na()

# Visualize training data distributions
train_data_clean %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Box Plots of Training Set Variables", x = "Variable", y = "Value") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Fit Simple Linear Models
model_rh <- lm(precip ~ relative_humidity, data = train_data_clean)
model_temp <- lm(precip ~ dry_bulb_temp_f, data = train_data_clean)
model_wind <- lm(precip ~ wind_speed, data = train_data_clean)
model_pressure <- lm(precip ~ station_pressure, data = train_data_clean)

# Function to visualize models
plot_model <- function(data, x_var, y_var, model, x_label, y_label, title) {
  ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "lm", col = "red", se = FALSE) +
    labs(title = title, x = x_label, y = y_label) +
    theme_minimal()
}

# Plot individual models
plot_model(train_data_clean, "relative_humidity", "precip", model_rh, "Relative Humidity", "Precipitation", "Precip vs Relative Humidity")
plot_model(train_data_clean, "dry_bulb_temp_f", "precip", model_temp, "Dry Bulb Temp (F)", "Precipitation", "Precip vs Dry Bulb Temp")
plot_model(train_data_clean, "wind_speed", "precip", model_wind, "Wind Speed", "Precipitation", "Precip vs Wind Speed")
plot_model(train_data_clean, "station_pressure", "precip", model_pressure, "Station Pressure", "Precipitation", "Precip vs Station Pressure")

# Fit Combined Linear Model
model_combined <- lm(precip ~ relative_humidity + dry_bulb_temp_f + wind_speed + station_pressure, data = train_data_clean)
summary(model_combined)

# Fit Polynomial Model
model_poly <- lm(precip ~ poly(station_pressure, 2) + relative_humidity + wind_speed, data = train_data_clean)
summary(model_poly)

# Lasso Regression with Cross-Validation
lasso_spec <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")

lasso_workflow <- workflow() %>%
  add_model(lasso_spec) %>%
  add_formula(precip ~ relative_humidity + dry_bulb_temp_f + wind_speed + station_pressure)

set.seed(123)
cv_folds <- vfold_cv(train_data_clean, v = 5)

lasso_tune <- tune_grid(
  lasso_workflow,
  resamples = cv_folds,
  grid = 20,
  metrics = metric_set(rmse, rsq)
)

# Select and finalize best Lasso model
best_lasso <- select_best(lasso_tune, metric = "rmse")
final_lasso <- finalize_workflow(lasso_workflow, best_lasso)
final_lasso_fit <- fit(final_lasso, data = train_data_clean)

# Evaluate models on the test set
evaluate_model <- function(model, test_data, model_type = "lm") {
  if (model_type == "tidymodels") {
    predictions <- predict(model, new_data = test_data) %>%
      bind_cols(test_data)
  } else {
    predictions <- test_data %>%
      mutate(.pred = predict(model, newdata = test_data))
  }

  metrics <- predictions %>%
    metrics(truth = precip, estimate = .pred) %>%
    filter(.metric %in% c("rmse", "rsq"))

  return(metrics)
}

# Evaluate all models
metrics_rh <- evaluate_model(model_rh, test_data)
metrics_temp <- evaluate_model(model_temp, test_data)
metrics_wind <- evaluate_model(model_wind, test_data)
metrics_pressure <- evaluate_model(model_pressure, test_data)
metrics_combined <- evaluate_model(model_combined, test_data)
metrics_poly <- evaluate_model(model_poly, test_data)
metrics_lasso <- evaluate_model(final_lasso_fit, test_data, model_type = "tidymodels")

# Create comparison table
model_names <- c("Simple RH", "Simple Temp", "Simple Wind", "Simple Pressure",
                 "Combined Model", "Polynomial Model", "Lasso Model")

test_rmse <- c(
  metrics_rh$.estimate[metrics_rh$.metric == "rmse"],
  metrics_temp$.estimate[metrics_temp$.metric == "rmse"],
  metrics_wind$.estimate[metrics_wind$.metric == "rmse"],
  metrics_pressure$.estimate[metrics_pressure$.metric == "rmse"],
  metrics_combined$.estimate[metrics_combined$.metric == "rmse"],
  metrics_poly$.estimate[metrics_poly$.metric == "rmse"],
  metrics_lasso$.estimate[metrics_lasso$.metric == "rmse"]
)

test_rsq <- c(
  metrics_rh$.estimate[metrics_rh$.metric == "rsq"],
  metrics_temp$.estimate[metrics_temp$.metric == "rsq"],
  metrics_wind$.estimate[metrics_wind$.metric == "rsq"],
  metrics_pressure$.estimate[metrics_pressure$.metric == "rsq"],
  metrics_combined$.estimate[metrics_combined$.metric == "rsq"],
  metrics_poly$.estimate[metrics_poly$.metric == "rsq"],
  metrics_lasso$.estimate[metrics_lasso$.metric == "rsq"]
)

comparison_df <- data.frame(
  Model = model_names,
  Test_RMSE = test_rmse,
  Test_R_squared = test_rsq
)

# Print comparison table
print(comparison_df)

