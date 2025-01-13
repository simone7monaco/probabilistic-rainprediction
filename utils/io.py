import os
from pathlib import Path
from typing import Literal, AnyStr, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold  # type: ignore


# def get_model(topdir: Path, model_name: str, date: str, case_study_max: float) -> np.ndarray:
#     file_path = topdir/"models"/f"{model_name}_{date}_0024_regrid.csv"
#     if not file_path.exists():
#         file_path = topdir/"models"/"data_intenseEvents"/"new_dataset"/f"{model_name}_{date}_0024_regrid.csv"
#     assert file_path.exists(), f"File {file_path} does not exist"
#     model_data = pd.read_csv(file_path, sep=";", header=None).to_numpy()
#     return model_data / case_study_max


def get_mask_indices(topdir: str, ispadded:bool):
    cs = topdir.stem.split('_')[-1] # OI | radar
    file_path = next(topdir.glob(f"{cs}*regrid*piem_vda.csv"))
    if not file_path.exists():
        file_path = next(topdir.glob(f"{cs}*piem_vda.csv"))
    # file_path = os.path.join(topdir, "OI_regrid_mask_piem_vda_unet.csv")
    
    mask = pd.read_csv(file_path, sep=";", header=None).to_numpy()
    if ispadded:
        mask = np.pad(mask, ((0, 0), (0, 128-mask.shape[1])), mode='constant', constant_values=0)

    indices = np.where(mask == 1)
    indices_zero = np.where(mask == 0)
    return indices, indices_zero, mask


# def get_obs(topdir: str, date: str, case_study_max: float) -> np.ndarray:
#     # file_path = topdir / "obs" / "data" / f"OI_{date}_regrid.csv"
#     case_study = topdir.stem.split('_')[-1] # OI | radar
#     file_path = topdir / "obs" / "data"/f"{case_study}_{date}_regrid.csv"
#     if not file_path.exists():
#         file_path = file_path.parents[1]/"data_intenseEvents"/"new_dataset"/f"{case_study}_{date}_regrid.csv"
#         if not file_path.exists():
#             raise FileNotFoundError(f"File '{case_study}*{date}*.csv' does not exist")
#     obs_data = pd.read_csv(file_path, sep=";", header=None).to_numpy()
#     return obs_data / case_study_max


def load_data(
    topdir: Path,
    dates: np.ndarray,
    case_study_max: float,
    indices: np.ndarray,
    indices_zero: np.ndarray,
    available_models: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    models_data = []
    obs_data = []

    dates_df = pd.read_csv(topdir / "allevents_dates.csv", sep=";", index_col=0)
    for date in dates:
        _models_tmp = []
        #print(date)
        for model_name in available_models:
            _mdl_data = pd.read_csv(topdir / dates_df.loc[date, model_name.upper()],
                                    sep=";", header=None).to_numpy() / case_study_max
            
            _mdl_data=np.hstack((_mdl_data,np.zeros((_mdl_data.shape[0], int(2**np.ceil(np.log2(_mdl_data.shape[1]))) - _mdl_data.shape[1])))) # **update:** this was hardcoded as (96, 12) [12 supposed to be 128]
            _mdl_data[indices_zero] = 0

            if _mdl_data.shape[0] % 2 != 0:
                _mdl_data = _mdl_data[:-1]
            _models_tmp.append(_mdl_data)
        models_data.append(_models_tmp)
        _obs_data = pd.read_csv(topdir / dates_df.loc[date, "OBS"],
                                sep=";", header=None).to_numpy() / case_study_max
        _obs_data=np.hstack((_obs_data,np.zeros((_obs_data.shape[0], int(2**np.ceil(np.log2(_obs_data.shape[1]))) - _obs_data.shape[1]))))
        _obs_data[indices_zero] = 0
        if _obs_data.shape[0] % 2 != 0:
            _obs_data = _obs_data[:-1]
             
        obs_data.append(_obs_data)
    x = np.stack(models_data, axis=0).astype(np.float32)
    y = np.stack(obs_data, axis=0).astype(np.float32)
    return x, y, x.shape[1],1
        

def get_casestudy_stuff(input_path:str, n_split: int, case_study:str, ispadded:bool, seed:int):
    case_study_max=483.717752
    available_models = ["bol00", "e1000", "c2200", "c5m00"]

    dates = pd.read_csv(input_path / "allevents_dates.csv", sep=";")
    skf = StratifiedKFold(n_splits=9, random_state=seed, shuffle=True)
    train_index, test_index = list(skf.split(dates, dates.NAME))[n_split]
    val_index, train_index = np.split(train_index, [len(test_index)])
    train_dates = dates.iloc[train_index].DATA.values
    val_dates = dates.iloc[val_index].DATA.values
    test_dates = dates.iloc[test_index].DATA.values

    indices_one, indices_zero, mask = get_mask_indices(input_path, ispadded)
    nx, ny = mask.shape
    return case_study_max, available_models, train_dates, val_dates, test_dates, indices_one, indices_zero, mask, nx, ny



def date_features(dates:List[AnyStr]):
    """
    Extract temporal features from a date expressed in the form yyyymmdd (e.g. 20200612)
    """
    df = pd.to_datetime(dates, format='%Y%m%d').rename('date').to_frame()
    season = df.date.dt.month.apply(lambda x: 0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3)
    df['sin_season'] = np.sin(2*np.pi*season/4)
    df['cos_season'] = np.cos(2*np.pi*season/4)
    df['sin_month'] = np.sin(2*np.pi*df.date.dt.month/12)
    df['cos_month'] = np.cos(2*np.pi*df.date.dt.month/12)
    df['sin_week'] = np.sin(2*np.pi*df.date.dt.isocalendar().week/52)
    df['cos_week'] = np.cos(2*np.pi*df.date.dt.isocalendar().week/52)
    df['sin_day'] = np.sin(2*np.pi*df.date.dt.dayofyear/365)
    df['cos_day'] = np.cos(2*np.pi*df.date.dt.dayofyear/365)

    return df.reset_index(drop=True).drop(columns=['date']).values.astype(np.float32)