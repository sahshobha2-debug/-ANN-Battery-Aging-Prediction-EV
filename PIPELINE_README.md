# ANN Battery SOH Pipeline — Methodology Map


|---|---|---|
| 1 | Data collection (input features / output target) | `Step01_DataCollection.m` |
| 2a | Handle missing values | `Step02_DataPreprocessing.m` |
| 2b | Remove duplicates | `Step02_DataPreprocessing.m` |
| 2c | Detect outliers (z-score, IQR, Isolation Forest) | `Step02_DataPreprocessing.m` |
| 2d | Standard normalization | `Step02_DataPreprocessing.m` |
| 3 | Correlation heat map / redundancy | `Step03_CorrelationHeatmap.m` |
| 4 | PCA | `Step04_PCA.m` |
| 5 | Feature selection (PC scores OR selected features) | `Step05_FeatureSelection.m` |
| 6 | Split dataset (train/val/test) | `Step06_DataSplit.m` |
| 7 | Hyperparameter tuning (activation, hidden size, learning rate) | `Step07_HyperparameterTuning.m` |
| 8 | Initialize weights | `trainManualNN.m` (called by Step 8) |
| 9 | Forward propagation | `trainManualNN.m` |
| 10 | Loss (MSE for regression) | `trainManualNN.m` |
| 11 | Backpropagation (chain rule) | `trainManualNN.m` |
| 12 | Update weights (Adam) | `trainManualNN.m` |
| 13 | Validate on validation set | `trainManualNN.m` |
| 14 | Repeat epochs | `trainManualNN.m` |
| 15 | Hyperparameter tuning (revisited) | `Step07_HyperparameterTuning.m` |
| 16 | Early stopping | `trainManualNN.m` |
| 17 | Metrics: RMSE, MAE, MSE, R² | `Step09_EvaluationMetrics.m` |

## Important notes to mention to your supervisor

**Why a from-scratch network instead of `fitnet`?**
Your original code trains with MATLAB's `fitnet` + `trainlm`, which is
the Levenberg-Marquardt algorithm — a good, standard choice, but it is
**not** Adam, and MATLAB's shallow-network toolbox (`fitnet`) doesn't
expose a literal Adam option. Since the methodology sheet specifically
asks for the Adam update rule and explicit forward/backprop steps,
`trainManualNN.m` implements a single-hidden-layer network by hand —
every weight initialization, forward pass, loss calculation, gradient,
and Adam update is visible as plain MATLAB code with a comment tying
it to the matching methodology number. This is a good "show your
understanding of the math" artifact to present alongside the result.

Your original `fitnet`/`trainlm` pipeline (in `charge_cycle.m`) is
still valid — it's a legitimate, well-tested production model that
already gave you strong results. Nothing here throws that away; think
of the two as: `fitnet`/`trainlm` = the practical model you'd actually
deploy, `trainManualNN.m` = the transparent demonstration of what's
happening mathematically underneath any neural network, Adam-based or
otherwise.

**On missing-value handling (Step 2a):**
Removing rows with no impedance reading (as instructed) drops every
early-experiment cycle, since NASA only started running impedance
tests partway through each battery's life. This is a real trade-off —
worth a sentence in your report acknowledging it, in case your
supervisor asks why the dataset shrank.

**On Isolation Forest (Step 2c):**
`iforest` requires MATLAB R2021b+ with Statistics and Machine Learning
Toolbox. If it's unavailable, the script automatically falls back to
z-score + IQR only and prints a warning — the pipeline still runs.

**On feature selection (Step 5):**
The `USE_PCA_SCORES` flag lets you produce results both ways (PC
scores vs. selected raw features) with one line changed — useful if
your supervisor wants to see both compared side by side.

## Files you'll need in your MATLAB working folder
`B0005.mat`, `B0006.mat`, `B0018.mat` (same NASA battery data files
your original scripts used) — place them in the same folder as these
scripts before running `Step01_DataCollection.m`.
