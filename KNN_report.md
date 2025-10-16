# KNN Model for Used Car Price Prediction

Before applying the KNN model, we performed several data cleaning steps to ensure a stable and reliable distance calculation. Since KNN is highly sensitive to the choice of distance metric, only essential preprocessing steps were applied at this stage. Specifically, we dropped records with missing values in key columns, removed price outliers using the IQR method, and trimmed whitespace in categorical fields.

To make it easier to address the research questions, we also engineered new numerical features to better capture depreciation patterns. For example, we normalized mileage by age to create the `Mileage_per_year` feature, and applied logarithmic transformations to mileage and age to account for non-linear effects. Additionally, categorical features were encoded using `OneHotEncoder` with `min_frequency = 20`, which merges rare categories and reduces noise in high-dimensional space.

We chose the **K-Nearest Neighbors (KNN)** algorithm because it is a non-parametric method that relies on local neighborhood structure, making it well-suited to capture brand-driven local price patterns without assuming a specific functional form. To further stabilize the model, we used **L1 distance (Manhattan distance)**, which is more robust in high-dimensional sparse spaces, and applied distance weighting so that closer neighbors have a greater influence on the prediction.

As with other non-neural machine learning models, we split the dataset into 80% training data and 20% test data, with a random seed of 42 to ensure reproducibility. We then applied cross-validation to determine the optimal number of neighbors that minimizes the Mean Absolute Error (MAE). Finally, we evaluated the model using **MAE**, **RMSE**, and **R²**, which together provide a comprehensive assessment of prediction accuracy and model performance.

Although the K-Nearest Neighbors (KNN) algorithm is intuitive and effective for capturing local patterns, it has several notable limitations.

First, KNN suffers from the curse of dimensionality. When the number of features is high, the distance between samples tends to become similar, which weakens the model’s ability to identify truly similar neighbors. In our case, after applying `OneHotEncoder` to encode categorical variables like brand and model, the feature space expands rapidly. Even with the L1 distance metric, this problem cannot be fully avoided.

Second, KNN is very sensitive to feature scaling and noisy data. If features are not normalized properly, some variables may dominate the distance computation. Additionally, outliers or incorrect values can easily distort predictions.

Finally, KNN has no explicit training phase, meaning that all computations occur during the prediction stage. For each prediction, the algorithm must calculate the distance between the input and every training sample, which can become extremely time-consuming as the dataset grows. This makes real-time prediction less efficient compared to models that learn a compact representation during training.

In our case, when applying KNN to this dataset, the prediction process took at least three minutes to complete, highlighting the algorithm’s high computational cost at inference time.

## Model Performance

- **MAE**: £179,497.13
- **RMSE**: £296,740.75
- **R²**: 0.9263

These results indicate that the KNN model provides strong predictive performance, achieving an R² of approximately 0.93, suggesting that local neighborhood information—particularly brand-related features—offers substantial explanatory power for used car pricing. However, the relatively long prediction time underscores one of KNN’s key limitations in terms of computational efficiency and scalability.

---

## 1. What models retain their value over higher mileage?

Based on Figure 1, we can observe a clear negative relationship between mileage and car price across all brands — as mileage increases, the price of the vehicle decreases. However, the rate of depreciation varies significantly between brands. For example, while Volkswagen vehicles also show a price decline with increasing mileage, their prices remain consistently higher than brands such as Nissan or Lada at comparable mileage levels. In some mileage ranges, Volkswagen vehicles are priced significantly above other brands, indicating stronger brand value retention.

This observation is further supported by Figure 2, which illustrates the brand value retention rate. A higher retention rate indicates that customers are more willing to pay a higher price for high-mileage and older second-hand cars of that brand. For instance, brands with higher retention rates maintain stronger resale value even as mileage increases.

Taken together, these two figures suggest that while price depreciation is a universal trend across all brands, the extent of depreciation differs. This difference is influenced not only by the brand itself but also by specific car models and brand positioning. Consequently, different brands exhibit distinct price depreciation patterns over mileage, which makes brand an important factor to include in predictive modeling of used car prices.

---

## 2. What factors have the strongest influence on car price?

To better understand the relationship between engine horsepower and price, we generated a scatterplot of **Price vs Engine HP** and applied **LOWESS (Locally Weighted Scatterplot Smoothing)** to visualize the overall trend.

Before plotting, the data was lightly clipped to the 1st and 99th percentiles for both horsepower and price to reduce the impact of extreme outliers. This ensures that the trend line is not overly skewed by a few high-end vehicles. The scatterplot was plotted with a low alpha value for better visual clarity in high-density regions, and the LOWESS curve was overlaid in red to show the smoothed price trend.

The resulting plot (Figure 3) reveals a strong, non-linear positive relationship between horsepower and price. Price increases rapidly between approximately 50 and 200 HP, then continues to grow at a slower pace beyond 200 HP. This pattern reflects diminishing marginal returns of engine power on price, which aligns with real-world vehicle pricing structures where high-performance vehicles command a premium but have a smaller incremental price increase at the top end.

This visualization provides direct evidence that **engine horsepower** is one of the most influential numerical features affecting vehicle price in the dataset. The smooth and monotonic trend makes this feature particularly well-suited for non-parametric models such as the K-nearest neighbors algorithm, which rely on local neighborhood similarity rather than a predefined functional form.