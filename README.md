# ML Portfolio

A collection of short machine-learning notebooks covering regression, classification, clustering, time-series
forecasting, deep learning (CNNs, PyTorch basics) and computer vision. Each notebook stands alone.

## Projects

| Notebook | Topic | What it does | Data |
|---|---|---|---|
| [Fashion_MNIST_CNN.ipynb](Fashion_MNIST_CNN.ipynb) | Deep learning · image classification | Two-block CNN in TensorFlow/Keras that classifies the 10 Fashion-MNIST clothing classes; confusion matrix, classification report, ROC curves | Downloaded automatically by Keras |
| [Age_and_Gender_Detection.ipynb](Age_and_Gender_Detection.ipynb) | Computer vision | Detects faces in a webcam photo with OpenCV's DNN face detector and predicts gender and an age bracket with pretrained Caffe models | Model weights from Google Drive (auto-download) |
| [Crypto_Analysis.ipynb](Crypto_Analysis.ipynb) | Regression · time series | Models Ethereum's daily close with XGBoost and forecasts the last 30 days with ARIMA(2,1,1) | `ETH-USD.csv` (Yahoo Finance) |
| [Decision_tree_linear.ipynb](Decision_tree_linear.ipynb) | Regression | Decision-tree regressor that predicts total hospital charges from age, sex, length of stay and diagnosis group, with tree plot | `linear_regression_dataset.csv` |
| [Decision_tree_logistic.ipynb](Decision_tree_logistic.ipynb) | Classification | Pruned decision-tree classifier that predicts `RainTomorrow` from Australian daily weather observations, with tree plot | `weatherAUS_dataset.csv` |
| [Clustering.ipynb](Clustering.ipynb) | Unsupervised learning | K-means (k = 4) on standardised weather features, PCA to 2-D for plotting, elbow curve for k = 1…9 | `weather_dataset_final.csv` |
| [Linear_regression_model.ipynb](Linear_regression_model.ipynb) | Deep-learning fundamentals | Linear regression from scratch in PyTorch: autograd, MSE loss, manual gradient descent, then the same with `nn.Linear` + `SGD` | Small inline toy dataset |

## Results (as shown in the notebooks)

| Notebook | Result |
|---|---|
| Fashion_MNIST_CNN | **90.49%** test accuracy after 5 epochs; macro F1 0.904. Weakest class: Shirt (F1 0.72). Re-executed for this README with TF 2.21 and a fixed seed. The original Colab run had 90.74%. |
| Crypto_Analysis | XGBoost RMSE **105.19 USD** on a random 20% hold-out. The ARIMA cell was fixed and needs re-running for its MAE (see Notes). |
| Linear_regression_model | MSE on the toy data falls from 9001.89 (random weights) to 431.04 after 100 manual gradient-descent epochs. |
| Clustering | 55,911 complete rows clustered into 4 groups; PCA scatter and elbow plot saved in the notebook. |
| Age_and_Gender_Detection | Qualitative: annotated webcam frame (face box + `gender, age-range` label). |
| Decision_tree_linear / Decision_tree_logistic | Outputs cleared after bug fixes (see Notes); re-run with the datasets to regenerate. |

## Datasets

The CSV files are **not** in the repository. Put them next to the notebooks under the file names below.

| File name used | Source / how to get it |
|---|---|
| `weatherAUS_dataset.csv` | Kaggle [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) (`weatherAUS.csv`, renamed). |
| `weather_dataset_final.csv` | A pre-processed version of the same weather data in which the categorical columns (`Location`, wind directions, `RainToday`) are already numeric, since K-means needs numeric input. The exact preprocessing is not in the repo. Label-encoding those columns (as in `Decision_tree_logistic.ipynb`) reproduces the expected format. |
| `linear_regression_dataset.csv` | Hospital discharge cost data with columns `AGE, FEMALE, LOS, RACE, TOTCHG, APRDRG` (the widely used "Hospital Costs" teaching dataset derived from HCUP). Not included. |
| `ETH-USD.csv` | Yahoo Finance [ETH-USD history](https://finance.yahoo.com/quote/ETH-USD/history/), daily, from 2017-11-09. Set `DATA_PATH` in the notebook. |
| Fashion-MNIST | Downloaded by `tf.keras.datasets.fashion_mnist.load_data()`. |
| Age/gender model weights | `Age_and_Gender_Detection.ipynb` downloads them into `modelNweight/` from the author's Google Drive folder with `gdown` (about 90 MB): OpenCV SSD face detector plus the Levi & Hassner age and gender Caffe models. |

## How to run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

- On macOS, XGBoost needs the OpenMP runtime: `brew install libomp`.
- `Age_and_Gender_Detection.ipynb` takes the photo with the webcam on **Google Colab**. Run locally, it uses an image you save as `photo.jpg` and displays results with Matplotlib.
- Every notebook still works on Colab. Upload the CSV next to the notebook, or change the path.

## Notes and fixes in this version

- **Fashion_MNIST_CNN** (renamed from `MNIST (1).ipynb`):
  - Removed the `!pip` cell.
  - Updated for Keras 3 (`tf.keras.Input`, plus a shim so `visualkeras` can draw the model).
  - Replaced a precision/recall cell that misused binary Keras metrics (it reported ~0.98) with macro averages.
  - Removed an "IoU" calculation that is not meaningful for classification (it always gave 0.10).
  - Re-executed end to end.
- **Crypto_Analysis** (renamed from `Crypto_Analysis (1).ipynb`):
  - Replaced the Colab Drive path with `DATA_PATH`.
  - The ARIMA cell used `forecast(...)[0]`, which raises `KeyError` on pandas 3 and compared one forecast value with 30 test days. Its earlier MAE (336.93) was wrong, so that output was cleared.
  - The XGBoost model uses same-day High/Low with a random split, so its RMSE is not a forecast of future prices.
- **Decision_tree_linear**:
  - The model was trained on *all* rows, including the rows later used as the test set, so the previously printed R² of 0.970 was not a hold-out score. It now trains on rows 0–348 and tests on the rest.
  - Tree-plot feature names now match the model inputs.
  - Outputs cleared; re-run with the dataset.
- **Decision_tree_logistic**:
  - The tree plot was labelled with the wrong column names (`wth.columns` instead of the model's features); fixed.
  - `RainToday` is encoded if it is `Yes`/`No`.
  - Added `.copy()` to avoid `SettingWithCopyWarning`.
  - The tree-plot output was cleared, because it showed the wrong labels.
- **Age_and_Gender_Detection**: the `!gdown` cell pointed at a Drive *folder*, so it only downloaded an HTML page. It now uses `gdown.download_folder`, and the code falls back gracefully outside Colab.
- **Clustering** and **Linear_regression_model**: unchanged.

## Repository structure

```
.
├── Age_and_Gender_Detection.ipynb
├── Clustering.ipynb
├── Crypto_Analysis.ipynb
├── Decision_tree_linear.ipynb
├── Decision_tree_logistic.ipynb
├── Fashion_MNIST_CNN.ipynb
├── Linear_regression_model.ipynb
└── requirements.txt
```

## Tech stack

Python, Jupyter / Google Colab, pandas, NumPy, scikit-learn, Matplotlib, seaborn, Plotly, TensorFlow/Keras,
PyTorch, XGBoost, statsmodels, OpenCV.

## Licence

[MIT](LICENSE) © 2023–2026 Samarth Jaiswal
