Here's, Thes summary of what all i did, some of their code is missing so yeah----Not everything worked to be exact most of them didn't worked
but was great learning experience:

1- Feature Engineering: Created new features, like room_per_person, using existing features (e.g., total rooms and population). These worked well.

2- Outlier Removal: Removed 2,064 outliers using Isolation Forest and manually removed 300 more based on low district population or unrealistic       ratios. Used a color-mapped histogram to visualize outliers and assess the median housing price's relation to each feature.

3- Pipeline Development: Built custom transformers, including one for cluster similarity (which was effective) and created a processing pipeline to    handle scaling, imputation, log transformation (for skewed features), and other transformations.

4- Model Selection: Initial models like linear regression and SVR (linear and polynomial) performed poorly. Decision tree regressor overfitted and did not generalize well on validation data. Random forest and XGBoost regressors, however, performed well.

5- Hyperparameter Tuning: Used RandomizedSearchCV and Optuna for hyperparameter tuning, with Optuna yielding the best parameters.

6- Error Visualization and Analysis: Investigated error concentrations using scatter plots and KMeans clustering, identifying the southern coastline as an area of high error.

6- Error Diagnosis: Found significant underprediction errors in urban centers along the southern coast, likely due to overfitting and underfitting.

7- Urban Center Proximity: Added latitude and longitude of important urban centers, calculated haversine distances to each, and adjusted model weighting to reduce errors for southern urban center samples.

8- SHAP Analysis: Examined SHAP values to understand which features contributed most to overprediction and underprediction. This analysis was informative but did not directly reduce errors.

9- Separate Model for High-Error Regions: Created a separate model for the southern coastline to address high error rates in this region.

10- Findings: Concluded that errors in southern urban centers were partly due to missing features like education facilities, healthcare, entertainment, population density, and crime indices. The dataset was also limited in size—after removing outliers and the test set, only around 14,000 samples were left for training, which likely affected the model's performance.
