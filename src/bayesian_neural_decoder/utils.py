import numpy as np
import pandas as pd
import track_linearization as tl

def project_1D_position_to_2D(position_1D: float, 
                              place_bin_centers_df: pd.DataFrame) -> np.ndarray:
    idx = np.histogram(position_1D, bins=place_bin_centers_df["linear_position"])[0].argmax()
    return place_bin_centers_df.iloc[idx][["x_position", "y_position"]].to_numpy(dtype=float)

def get_posterior_2D(posterior: np.ndarray,
                     position_bins_2D: np.ndarray,
                     place_bin_centers_df: pd.DataFrame,
                     is_track_interior: np.ndarray,
                     band_width: int = 1) -> np.ndarray:
    
    # Copy the 2D position bins to avoid modifying the original array
    posterior_2D = position_bins_2D.copy()
    
    # Get the integer indices of the positions corresponding to the posterior probabilities
    position_idx = np.round(
        place_bin_centers_df.loc[is_track_interior][["x_position", "y_position"]].to_numpy(dtype=float), 
        0
    ).astype(int)

    # If band_width is less than 1, return the posterior probabilities at the position indices
    if band_width < 1:
        posterior_2D[position_idx[:,0], position_idx[:,1]] = posterior[is_track_interior]
        return posterior_2D
    
    # Get the dimensions of the 2D grid
    max_x, max_y = posterior_2D.shape

    # Create offsets within the band_width
    offsets = np.array([(dx, dy) 
                        for dx in range(-band_width, band_width + 1) 
                        for dy in range(-band_width, band_width + 1)])
    num_offsets = offsets.shape[0]
    
    # Expand positions by adding offsets to each position index
    expanded_positions = position_idx[:, np.newaxis, :] + offsets[np.newaxis, :, :]  # Shape: (num_positions, num_offsets, 2)
    expanded_positions = expanded_positions.reshape(-1, 2)  # Flatten to (num_positions * num_offsets, 2)
    
    # Repeat posterior probabilities for each offset
    posterior_probs = posterior[is_track_interior]
    posterior_probs_expanded = np.repeat(posterior_probs, num_offsets)
    
    # Filter out positions that are outside the bounds of the grid
    valid_indices = (
        (expanded_positions[:, 0] >= 0) & (expanded_positions[:, 0] < max_x) &
        (expanded_positions[:, 1] >= 0) & (expanded_positions[:, 1] < max_y)
    )
    
    positions = expanded_positions[valid_indices]
    posterior_probs_expanded = posterior_probs_expanded[valid_indices]
    
    # Assign posterior probabilities to the expanded positions
    # In case of overlapping positions, take the maximum probability
    posterior_2D_flat = posterior_2D.flatten()
    flat_indices = positions[:, 0] * max_y + positions[:, 1]
    np.maximum.at(posterior_2D_flat, flat_indices, posterior_probs_expanded)
    posterior_2D = posterior_2D_flat.reshape(posterior_2D.shape)
    
    return posterior_2D
    
def linearize_position(position_2D: np.ndarray, 
                       track_graph,
                       edge_order,
                       edge_spacing) -> float:
    position_df = tl.get_linearized_position(position_2D[np.newaxis], track_graph, edge_order, edge_spacing)
    return float(position_df["linear_position"].values[0])