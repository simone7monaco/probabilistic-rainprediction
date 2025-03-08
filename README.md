# Uncertainty-Aware Methods for Enhancing Rainfall Prediction with deep-learning based Post-Processing Segmentation

Simone Monaco, Luca Monaco and Daniele Apiletti

This repository contains the code and the data used in the paper, submitted to Computers & Geosciences. 


## Data

The data is available at the following link: [https://doi.org/10.5281/zenodo.14639278](https://doi.org/10.5281/zenodo.14639278). It is restricted but will be available upon acceptance.

Once downloaded, the data should be placed in the `data/` directory as follows:

```
data
└── 24h_10mmMAX_OI
    ├── allevents_dates.csv
    ├── models
    ├── study_area.csv
    ├── obs
    ├── OI_20152022_10mmMAX.csv
    ├── OI_raw_mask_piem_vda.csv
    ├── OI_regrid_mask_piem_vda.csv
    ├── OI_regrid_mask_piem_vda_unet.csv
    ├── OI_regrid_quota_unet.csv
    └── split
```

## Installation

Ensure you have Python installed (recommended version: 3.9 or later). We recommend using a virtual environment. Then, install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Experiments

To train and evaluate the models, run the following command:

```bash
python training.py
```

The different networks can be selected by changing the `network_model` flag. Available options are
- `unet`: The baseline deterministic U-Net model.
- `mc_dropout_unet`: The U-Net model with Monte Carlo Dropout.
- `mc_dropout_unet_ensemble`: The ensemble of U-Net models with Monte Carlo Dropout.
- `sde_unet`: The Stochastic Differential Equation U-Net model.

Run the following command to have a list of all the available options:

```bash
python training.py --help
```




## License

This repository is provided for academic and research purposes. For any other usage, please contact the authors.