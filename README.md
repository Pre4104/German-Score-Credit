# German-Score-Credit-Risk

# German Credit Risk — Loan Approval Classifier

A binary classification project predicting whether a loan applicant is a **good or bad credit risk**, using the [German Credit Risk dataset](https://www.kaggle.com/datasets/jumpingdino/german-credit-dataset) (Hofmann, 1994 / UCI Statlog). Built as an end-to-end ML pipeline — EDA, fairness checks, modeling, and cost-aware evaluation — applying the same workflow used in an earlier Titanic case study to a real lending decision.

## Problem

Banks must decide whether to approve a loan based on an applicant's profile. The two error types here aren't equally costly:

- **Approving a bad-credit applicant** → direct financial loss (default risk)
- **Rejecting a good-credit applicant** → missed business (opportunity cost)

The dataset's own documentation weights these at **5:1** (approving bad credit is 5x worse), so accuracy alone is the wrong metric to optimize for. The goal is a model that supports faster, more consistent, risk-aware lending decisions.

## Approach

1. **EDA** — examined approval rates across savings, employment, housing, credit history, and loan terms; verified two ambiguous columns (`status_and_sex`, `n_guarantors`) against documentation rather than assuming their meaning.
2. **Fairness check** — looked at approval-rate gaps by `status_and_sex` and `is_foreign_worker`, including a sample-size caveat where one subgroup (n=37) was too small to support a reliable disparity claim.
3. **Data prep** — no missing values; categorical features one-hot encoded (ordinal-looking columns deliberately *not* mapped to integers, since EDA showed they aren't monotonically related to approval); numeric features standardized (scaler fit on train only); 80/20 stratified split.
4. **Modeling** — Logistic Regression, Decision Tree, Random Forest, and KNN, trained on identical preprocessed data.
5. **Evaluation** — compared models on accuracy/F1, per-class recall (catching bad-credit applicants specifically), and a cost-weighted score reflecting the bank's actual 5:1 error asymmetry.

## Key Result

Ranking models by raw accuracy or F1-Score points to **Random Forest**. Re-ranking by the dataset's actual cost matrix (5x cost for approving bad credit, 1x for rejecting good credit) flips the recommendation to **Decision Tree** — it produces the lowest total business cost despite the weakest raw accuracy, because it makes the fewest high-cost approval mistakes.

| Model | Accuracy | F1-Score | Weighted Cost (5×FP + 1×FN) |
|---|---|---|---|
| Logistic Regression | 0.710 | 0.797 | 186 |
| **Decision Tree** | 0.680 | 0.766 | **180** |
| Random Forest | 0.745 | 0.828 | 187 |
| KNN | 0.730 | 0.819 | 198 |

**Takeaway:** the "best" model depends on what the bank actually cares about — model selection should follow the business cost structure, not accuracy in isolation.

## Responsible AI Notes

- `is_foreign_worker` was kept in the model rather than dropped, since removing a sensitive feature doesn't remove the risk of proxy discrimination through correlated variables — the disparity is flagged with its sample-size caveat instead of acted on naively.
- `status_and_sex` shows a real 8–13 point approval-rate spread but conflates two attributes into one column, limiting how precisely the disparity can be isolated.
- Neither finding is conclusive on its own; a production deployment would need a formal fairness audit with a larger, better-balanced sample before this model informs real lending decisions.

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn kagglehub
python loan_approval_capstone.py
```

The script downloads the dataset via `kagglehub`, runs EDA prints, trains all four models, and outputs evaluation metrics, confusion matrices, and the cost-weighted comparison.

## Files

- `loan_approval_capstone.py` — full pipeline: data loading, EDA, preprocessing, modeling, evaluation
- `Capstone Project - German Credit.docx` — full written report with detailed EDA findings and fairness analysis
