# Customer Churn Prediction: BCG X Data Science Job Simulation

A churn analysis completed as part of the [BCG X Data Science job simulation on Forage](https://www.theforage.com/simulations/bcg/data-science-ccdz). The client, PowerCo, is a fictional gas and electricity supplier losing SME customers. The goal was to test whether **price sensitivity** drives churn and to build a model that predicts which customers are likely to leave.

> The datasets and starter notebooks were provided by Forage as part of the simulation. All analysis, feature engineering and write-ups are my own.

## Key findings

- **~10% of customers churned**, so the data is heavily imbalanced
- **Price is a weak driver of churn.** Price-based features ranked low in importance; consumption and profitability were the strongest predictors
- **The baseline model is not fit for use yet.** It has 90% accuracy and 81% precision, but only **4.6% recall**: it caught 17 of 366 churners in the test set. Its accuracy barely beats predicting that nobody churns (90.0%)
- **Recommendation:** hold off on the proposed blanket 20% discount, improve the model's recall, and investigate non-price drivers

## Project structure

| File | Description |
|---|---|
| `1_eda.ipynb` | Exploratory data analysis: data types, hidden missing values, churn rates by sales channel, distributions |
| `2_feature_engineering.ipynb` | Time-based features (tenure, months to contract end), price-change features, encoding, log transforms, removing redundant columns |
| `3_modelling.ipynb` | Random forest classifier, evaluation with precision, recall and F1, feature importance |
| `Executive_Summary_PowerCo.pdf` | One-slide executive summary for client stakeholders |
| `Data Description.pdf` | Column definitions provided by the simulation |
| `*.csv` | Datasets provided by the simulation |

## Approach

1. **Business understanding:** framed the price-sensitivity hypothesis and outlined the data needed
2. **EDA:** found that missing values were stored as the string `"MISSING"` (about 25% of `channel_sales`), that the largest sales channel had above-average churn, and that consumption columns were heavily right-skewed
3. **Feature engineering:** converted raw dates into durations, built features capturing price jumps and volatility over the year, one-hot encoded categoricals, log-transformed skewed columns and dropped a near-duplicate margin column (correlation 0.9999)
4. **Modelling:** trained a random forest (1,000 trees) and evaluated it with precision and recall rather than accuracy, because of the class imbalance
5. **Communication:** summarised findings and recommendations in a one-slide executive summary

## Next steps

- Address class imbalance (`class_weight='balanced'` or oversampling churners)
- Lower the decision threshold to trade some precision for higher recall
- Tune hyperparameters and compare against other models

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter
