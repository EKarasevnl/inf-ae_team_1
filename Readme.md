# RecSysCourse: inf-ae_team1


This document provides a complete walkthough of our work done for the **Recommender System Project: Fairness–Driven Infinite–Width
Autoencoder for Recommender Systems**. The repository is the altered version provided by the Teaching Team.


---

## 🧱 Project Structure

```bash
.
├── FairDiverse/              # FairDiverse repository directory
├── REPRO.md                  # Reproducibility instructions
├── Readme.md                 # Main project README
├── Readme_original_paper.md  # README from the original paper repo
├── RecBole/                  # RecBole baseline code and configs
├── baseline.py               # RecBole baseline runner
├── baseline_runs.job         # Job script for baseline runs
├── build_dataset.py          # Script for building datasets
├── data/                     # Raw and processed datasets
├── data.py                   # Data loading and processing
├── eval.py                   # Evaluation metrics and logic
├── hyper_params.py           # All model/configuration hyperparameters
├── jobs/                     # Job scripts for experiments and baselines
├── main.py                   # Main entry point for ∞-AE
├── metrics.py                # Metrics computation
├── model.py                  # Infinite-width autoencoder model
├── preprocess.py             # Dataset preprocessing
├── requirements.txt          # Python dependencies
├── utils.py                  # Utility functions
```

## 📝 What's changed:

This repository builds on the original ∞-AE codebase with several improvements for fairness, diversity, and reproducibility. Here's what's new and how to use each feature:

### 1. Reproducibility (provided by the TA)
- **What:** All steps to fully reproduce our results are provided.
- **How:** Simply follow the instructions in [`REPRO.md`](REPRO.md) for environment setup, data preparation, and running experiments.

### 2. Expanded Hyperparameters & Preprocessing
- **What:** The `hyper_params.py` file now includes more options for controlling preprocessing and fairness/diversity experiments.
- **How:** Edit `hyper_params.py` to:
  - Enable GINI or MMF metrics use (`use_gini`, `use_mmf`)
  - Set regularization strengths (`gini_reg`, `mmf_reg`)
  - Specify item/category columns and filtering (`item_id`, `category_id`, `categories_to_retain`)

### 3. Fairness & Diversity Metrics (GINI, MMF)
- **What:** The evaluation pipeline now computes GINI and MMF metrics to measure fairness and diversity in recommendations.
- **How:** Set `use_gini` and/or `use_mmf` to `True` in `hyper_params.py`.

### 4. RecBole Baselines with Fairness Metrics [TODO]
- **What:** RecBole baseline scripts are updated to also compute GINI and MMF metrics for fair comparison.
- **How:** Use the job scripts in the `jobs/` directory, especially `jobs/run_recbole.job`, to run RecBole baselines with the new metrics. [TODO] Point to the specific jobs

### 5. In-Processing Regularization for Fairness/Diversity
- **What:** The model now supports in-processing regularization for fairness/diversity via `mmf_reg` and `gini_reg`.
- **How:** Set these values in `hyper_params.py` to control the strength of fairness/diversity regularization during training.

### 6. FairDiverse-Compatible Output
- **What:** The pipeline can output results in a format compatible with the FairDiverse reranking toolkit.
- **How:**
  - Set `post_process = True` in `hyper_params.py` before running your experiment.
  - After running, use the generated files with FairDiverse an the input for reranking as described in [fairdiverse_tutorial.ipynb](https://github.com/XuChen0427/FairDiverse/blob/master/fairdiverse_tutorial.ipynb) (section 3.2: Withoout Input from FairDiverse).






## 📘 **Quick Reference Table:**

| Goal                              | What to Edit/Run                | Notes                                         |
|------------------------------------|----------------------------------|-----------------------------------------------|
| Reproducibility                    | `REPRO.md`                       | Full step-by-step instructions                |
| More hyperparameters & preprocessing| `hyper_params.py`                | See new options for fairness/diversity        |
| Fairness/diversity regularization  | `gini_reg`, `mmf_reg` in `hyper_params.py` | Set to nonzero to enable                     |
| Run RecBole baselines              | `jobs/run_recbole.job`           | Or other scripts in `jobs/`                   |
| FairDiverse compatibility          | `post_process = True` in `hyper_params.py` | Generates files for FairDiverse reranking     |



## 📦 Dependencies / References

This project repository uses the following frameworks / refers to the following papers:

- [github repository](https://github.com/noveens/infinite_ae_cf)
- [paper arxiv](https://arxiv.org/abs/2206.02626)

The original authors citation:

```
@article{inf_ae_distill_cf,
  title={Infinite Recommendation Networks: A Data-Centric Approach},
  author={Sachdeva, Noveen and Dhaliwal, Mehak Preet and Wu, Carole-Jean and McAuley, Julian},
  booktitle={Advances in Neural Information Processing Systems},
  series={NeurIPS '22},
  year={2022}
}
```
