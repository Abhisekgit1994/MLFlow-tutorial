## MLflow end-to-end:

1. Train a linear regression model

2. Package the code that trains the model in a reusable and reproducible model format

3. Deploy the model into a simple HTTP server that will enable you to score predictions

4. This tutorial uses a dataset to predict the quality of wine based on quantitative features like the wine’s “fixed acidity”, “pH”, “residual sugar”, and so on.
### Training the Model
First, train a linear regression model that takes two hyper parameters: alpha and l1_ratio.