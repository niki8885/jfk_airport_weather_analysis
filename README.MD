
# Weather Data Modeling Project

This repository contains an R script for analyzing and modeling weather data from JFK Airport. The goal of this project is to build and evaluate various regression models to predict precipitation (`precip`) based on several weather-related predictors.

## Files
- **`weather_modeling.R`**: The main R script for data preprocessing, modeling, and evaluation.
- **`jfk_weather_sample.csv`**: The dataset used for analysis.
- **`README.md`**: Documentation for the project.

## Project Overview
The analysis involves the following steps:
1. **Data Cleaning**:
    - Handle special cases in the `precip` variable (e.g., trace values marked as "T").
    - Remove missing values.

2. **Data Visualization**:
    - Visualize the distribution of predictors and response variables using box plots.

3. **Modeling**:
    - Fit various regression models:
        - Simple Linear Models (SLM) for individual predictors.
        - Combined Multiple Linear Model.
        - Polynomial Regression Model.
        - Lasso Regression with cross-validation.

4. **Model Evaluation**:
    - Evaluate model performance on the **testing set** using metrics like:
        - Root Mean Square Error (RMSE)
        - R-squared (R²)

5. **Model Comparison**:
    - Compare the models to identify the best-performing one.

## Dependencies
The following R packages are required to run the script:
- `tidymodels`
- `yardstick`
- `broom`
- `dplyr`
- `tidyverse`
- `rsample`
- `ggplot2`
- `glmnet`

To install them, run:
```r
install.packages(c("tidymodels", "yardstick", "broom", "dplyr", "tidyverse", "rsample", "ggplot2", "glmnet"))
```

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/niki8885/jfk_airport_weather_analysis.git
   cd weather-modeling
   ```

2. Open the `weather_modeling.R` script in your preferred R environment (e.g., RStudio).

3. Run the script to:
    - Preprocess the data.
    - Fit and evaluate various regression models.
    - Compare model performance.

4. View the `comparison_df` table for a summary of model performance metrics.

## Results
The script outputs a comparison table with RMSE and R-squared values for each model, helping you identify the best model for predicting precipitation.

## Example Output
| Model              | Test RMSE | Test R-squared |
|--------------------|-----------|----------------|
| Simple RH          | 0.045     | 0.062          |
| Simple Temp        | 0.050     | 0.040          |
| Simple Wind        | 0.047     | 0.055          |
| Simple Pressure    | 0.046     | 0.060          |
| Combined Model     | 0.043     | 0.080          |
| Polynomial Model   | 0.042     | 0.085          |
| Lasso Model        | 0.041     | 0.090          |

## License
This project is open source and available under the [MIT License](LICENSE).
