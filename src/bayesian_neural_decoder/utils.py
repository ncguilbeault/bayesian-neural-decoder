import numpy as np
import pandas as pd

def project_1D_position_to_2D(position_1D: float, 
                              place_bin_centers_df: pd.DataFrame) -> np.ndarray:
    idx = np.histogram(position_1D, bins=place_bin_centers_df["linear_position"])[0].argmax()
    return place_bin_centers_df.iloc[idx][["x_position", "y_position"]].to_numpy(dtype=float)

def get_posterior_2D(posterior: np.ndarray,
                     position_bins_2D: np.ndarray,
                     place_bin_centers_df: pd.DataFrame,
                     is_track_interior: np.ndarray) -> np.ndarray:
    posterior_2D = position_bins_2D.copy()
    position_idx = np.round(place_bin_centers_df.loc[is_track_interior][["x_position", "y_position"]].to_numpy(dtype=float), 0).astype(int)
    posterior_2D[position_idx[:,0], position_idx[:,1]] = posterior[is_track_interior]
    return posterior_2D