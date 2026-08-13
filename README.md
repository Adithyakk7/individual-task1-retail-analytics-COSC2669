# Individual Task 1 — Retail Commercial Analytics

COSC2669 — Individual Task 1, Part 1.3 (Data Analysis)

## What this is

Two ML models (Random Forest, Neural Network) run on two public retail
datasets, tied to the Commercial Data Analyst role at Coles Group I used
for Part 1.1. The idea is looking at retail commercial performance from
two angles — operational/store-level and customer-level — using the same
two models on both, so the results actually line up for comparison.

1. **Walmart Store Sales Forecasting** (Kaggle) — store/operational
   view, what drives weekly sales across stores and departments.
2. **Online Retail II** (UCI) — customer view, what separates high-value
   customers from low-value ones based on their purchase history (RFM).

Full write-up of what I found: [`outputs/insights_summary.md`](outputs/insights_summary.md)

## Folder structure

```
data/           raw datasets (not committed, see below for download links)
src/            standalone scripts for each stage of the pipeline
notebooks/      main notebook, already run with outputs saved in it
outputs/        cleaned data, model results, feature importances, write-up
```

## Getting the raw data

Didn't commit the raw data files to this repo since they're large and
easy enough to grab from the original sources:

1. **Walmart Store Sales Forecasting**
   https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data

   Grab `train.csv`, `features.csv`, `stores.csv` and put them in `data/`

2. **Online Retail II**
   https://archive.ics.uci.edu/dataset/502/online+retail+ii
   (no login needed)
   Grab `online_retail_II.xlsx` and put it in `data/`

## Running it

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/individual_task1_analysis.ipynb
```

Or just run the scripts one at a time:
```bash
cd src
python3 01_load_clean_walmart.py
python3 02_load_clean_retail.py
python3 03_model_and_evaluate.py
```

## Results

| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Walmart (store sales) | Random Forest | 0.758 | 0.510 | 0.821 | 0.629 | 0.858 |
| Walmart (store sales) | Neural Network | 0.814 | 0.669 | 0.509 | 0.578 | 0.852 |
| Online Retail II (customers) | Random Forest | 0.900 | 0.748 | 0.905 | 0.819 | 0.960 |
| Online Retail II (customers) | Neural Network | 0.907 | 0.879 | 0.728 | 0.796 | 0.951 |

See [`outputs/insights_summary.md`](outputs/insights_summary.md) for what
these actually mean.

## References

- Chen, D., Sain, S.L., & Guo, K. (2012). Data mining for the online
  retail industry: A case study of RFM model-based customer segmentation
  using data mining. *Journal of Database Marketing & Customer Strategy
  Management*, 19(3), 197–208.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). Learning
  representations by back-propagating errors. *Nature*, 323(6088), 533–536.
