# Explainable High-Value Retail Transaction Classification

A TensorFlow/Keras classification experiment using Integrated Gradients to interpret predictions of whether a retail transaction exceeds 500 in the dataset's monetary units.

## Business question
Explore whether product category, demographic attributes, and transaction timing contain predictive information about high-value transactions. This is not evidence of improved promotion or targeting outcomes.

## Technical approach
Date-derived month and weekday features, categorical encoding, standard scaling, a three-hidden-layer neural network with dropout, and an Integrated Gradients implementation.

## Saved results
- Test accuracy: 68.00%.
- Final training accuracy: 71.09%.
- Final validation accuracy: 65.00%.
- Mean absolute attribution ranking: Month, Gender, Age, DayOfWeek, Product_Category.

These are saved outputs from a single experiment, not newly reproduced results or a demonstrated improvement over a baseline.

## Explore and run
Open [the notebook](Assignment7.ipynb) in Colab, provide the Retail Sales CSV, replace the personal Drive path, and install its dependencies. The notebook cites Mohammad Talib's Retail Sales Dataset on Kaggle.

## Interpretation and limitations
Integrated Gradients measures input attribution; the network does not have an attention mechanism.
The saved comparison shows lower mean absolute attribution for Product_Category than Age. The original notebook prose describing the opposite direction needs correction.
Scaling is currently fitted before the test split and should be moved to a training-only pipeline.
The attribution baseline is derived from test observations; compare a documented training-derived baseline.
Feature comparisons use independent t-tests on paired observations; reassess the test design and multiple comparisons.

## Next steps
Add majority-class and Logistic Regression baselines, precision/recall/F1, ROC-AUC, reproducible seeds, and train-only preprocessing. Validate explanations before turning them into business recommendations.

## Project context
Developed by Yiren Shen for Duke's AIPI 590 Explainable AI course.
