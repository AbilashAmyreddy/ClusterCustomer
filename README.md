# ClusterCustomer

Customer segmentation dashboard built with Python and Streamlit. The app cleans transaction data, builds RFM features, loads a pre-trained K-Means model, and turns the results into interactive customer segments with business recommendations.

## What It Does

The dashboard analyzes the bundled Online Retail dataset or a custom CSV upload and surfaces:

* Customer-level RFM metrics: Recency, Frequency, and Monetary value
* K-Means cluster assignments using a saved model and scaler
* Segment naming and business recommendations generated from cluster behavior
* Interactive charts for 3D RFM space, PCA projection, distributions, and heatmaps
* Filterable data tables and CSV export for segmented customers

## Quick Start

1. Create and activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure the sample dataset exists at [data/OnlineRetail.csv](data/OnlineRetail.csv).
4. Generate the model artifacts if needed:

```bash
python train_model.py
```

5. Launch the dashboard from the active `.venv` environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

If `streamlit` is not recognized, use the `python -m streamlit run app.py` form instead of calling `streamlit` directly.


## Features

* Default analysis on the bundled Online Retail dataset
* Custom CSV upload with required columns validation
* Transaction cleaning for missing customer IDs, cancelled invoices, and invalid values
* RFM feature engineering and scaling
* Pre-trained K-Means segmentation
* Segment-level recommendations and summary metrics
* Visual analytics for RFM, PCA, cluster distribution, and heatmaps
* Data explorer with filtering, sorting, and CSV download

## Training And Validation

The project includes scripts for model building and validation:

* [train_model.py](train_model.py) trains the clustering model, evaluates candidate cluster counts, and saves `models/kmeans_model.pkl` and `models/scaler.pkl`
* [evaluation.py](evaluation.py) generates elbow and Davies-Bouldin plots under the generated `assets/` folder
* [segmentation.py](segmentation.py) loads the saved model and predicts cluster labels for new RFM data

## Project Structure

```text
ClusterCustomer/
├── app.py
├── data_loader.py
├── evaluation.py
├── feature_engineering.py
├── segmentation.py
├── visualizations.py
├── train_model.py
├── requirements.txt
├── README.md
├── assets/           # Generated charts and evaluation outputs
├── data/
│   └── OnlineRetail.csv
├── models/
└── src/
    └── utils.py
```

## Data Requirements

For custom uploads, the CSV must contain these columns:

* `InvoiceDate`
* `CustomerID`
* `Quantity`
* `UnitPrice`

The app also handles `InvoiceNo` when present so cancelled invoices can be removed.

## Model Workflow

1. Load and clean transaction data.
2. Compute customer-level RFM features.
3. Scale RFM values with `StandardScaler`.
4. Search for a suitable cluster count during training.
5. Fit K-Means and persist the model artifacts.
6. Load the saved model in the Streamlit app and generate clusters.
7. Derive segment names, recommendations, and dashboards from the cluster profiles.

## Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Matplotlib
* Seaborn
* Joblib

## Dataset

The bundled dataset is the classic Online Retail transaction dataset, which includes invoice information, stock codes, product descriptions, customer IDs, quantities, unit prices, and timestamps. It is suitable for RFM analysis and customer behavior segmentation.

## Notes

* Run `train_model.py` whenever you want to regenerate the saved model artifacts.
* The dashboard can work with the sample dataset immediately after the model files are available.
* Custom uploads are processed in-memory and do not overwrite the bundled dataset.

## Credit

Built by Abilash Amyreddy.

 

