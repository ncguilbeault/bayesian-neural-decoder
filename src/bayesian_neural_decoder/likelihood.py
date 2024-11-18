from replay_trajectory_classification.core import scaled_likelihood
import replay_trajectory_classification.likelihoods.multiunit_likelihood as ml 
import replay_trajectory_classification.likelihoods.multiunit_likelihood_gpu as mlgpu
import replay_trajectory_classification.likelihoods.multiunit_likelihood_integer_gpu as mligpu
from replay_trajectory_classification.likelihoods.spiking_likelihood_kde import poisson_log_likelihood

import numpy as np
import cupy as cp

def spiking_likelihood_kde(spikes, conditional_intensity, is_track_interior):

    log_likelihood = 0
    for spike, ci in zip(spikes, conditional_intensity.T):
        log_likelihood += poisson_log_likelihood(spike[np.newaxis], ci)

    mask = np.ones_like(is_track_interior, dtype=float)
    mask[~is_track_interior] = np.nan
    
    likelihood = scaled_likelihood(log_likelihood * mask)
    likelihood = likelihood[:, is_track_interior].astype(float)

    return likelihood

def multiunit_likelihood(multiunits, summed_ground_process_intensity, encoding_marks, encoding_positions, mean_rates, is_track_interior, interior_place_bin_centers, position_std, mark_std, interior_occupancy, n_track_bins):
    log_likelihood = -summed_ground_process_intensity * np.ones((1,1), dtype=np.float32)

    if not np.isnan(multiunits).all():
        # multiunit_idxs = np.where(~np.isnan(multiunits, axis=0))[0]

        for multiunit, enc_marks, enc_pos, mean_rate in zip(
                multiunits.T,
                encoding_marks,
                encoding_positions,
                mean_rates,
            ):
            is_spike = np.any(~np.isnan(multiunit))
            if is_spike:
                decoding_marks = np.asarray(
                    multiunit, dtype=np.float32
                )[np.newaxis]
                log_joint_mark_intensity = np.zeros(
                    (1, n_track_bins), dtype=np.float32
                )
                position_distance = ml.estimate_position_distance(
                    interior_place_bin_centers,
                    np.asarray(enc_pos, dtype=np.float32),
                    position_std,
                ).astype(np.float32)
                log_joint_mark_intensity[0] = ml.estimate_log_joint_mark_intensity(
                    decoding_marks,
                    enc_marks,
                    mark_std,
                    interior_occupancy,
                    mean_rate,
                    position_distance=position_distance,
                )
                log_likelihood[:, is_track_interior] += np.nan_to_num(
                    log_joint_mark_intensity
                )
    
    log_likelihood[:, ~is_track_interior] = np.nan
    likelihood = scaled_likelihood(log_likelihood)

    return likelihood

def multiunit_likelihood_gpu(multiunits, summed_ground_process_intensity, encoding_marks, encoding_positions, mean_rates, is_track_interior, interior_place_bin_centers, position_std, mark_std, interior_occupancy, n_track_bins):
    log_likelihood = -summed_ground_process_intensity * np.ones((1,1), dtype=np.float32)

    if not np.isnan(multiunits).all():
        # multiunit_idxs = np.where(~np.isnan(multiunits, axis=0))[0]

        for multiunit, enc_marks, enc_pos, mean_rate in zip(
                multiunits.T,
                encoding_marks,
                encoding_positions,
                mean_rates,
            ):
            is_spike = np.any(~np.isnan(multiunit))
            if is_spike:
                decoding_marks = cp.asarray(
                    multiunit, dtype=cp.float32
                )[cp.newaxis]
                log_joint_mark_intensity = np.zeros(
                    (1, n_track_bins), dtype=np.float32
                )
                position_distance = mlgpu.estimate_position_distance(
                    interior_place_bin_centers,
                    cp.asarray(enc_pos, dtype=cp.float32),
                    position_std,
                ).astype(cp.float32)
                log_joint_mark_intensity[0] = mlgpu.estimate_log_joint_mark_intensity(
                    decoding_marks,
                    enc_marks,
                    mark_std,
                    interior_occupancy,
                    mean_rate,
                    position_distance=position_distance,
                )
                log_likelihood[:, is_track_interior] += np.nan_to_num(
                    log_joint_mark_intensity
                )

                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()

    log_likelihood[:, ~is_track_interior] = np.nan
    likelihood = scaled_likelihood(log_likelihood)
    return likelihood

def multiunit_likelihood_integer_gpu(multiunits, summed_ground_process_intensity, encoding_marks, encoding_positions, mean_rates, is_track_interior, interior_place_bin_centers, position_std, mark_std, interior_occupancy, n_track_bins):
    log_likelihood = -summed_ground_process_intensity * np.ones((1,1), dtype=np.float32)

    if not np.isnan(multiunits).all():
        # multiunit_idxs = np.where(~np.isnan(multiunits, axis=0))[0]

        for multiunit, enc_marks, enc_pos, mean_rate in zip(
                multiunits.T,
                encoding_marks,
                encoding_positions,
                mean_rates,
            ):
            is_spike = np.any(~np.isnan(multiunit))
            if is_spike:
                decoding_marks = cp.asarray(
                    multiunit, dtype=cp.int16
                )[cp.newaxis]
                log_joint_mark_intensity = np.zeros(
                    (1, n_track_bins), dtype=np.float32
                )
                position_distance = mligpu.estimate_position_distance(
                    interior_place_bin_centers,
                    cp.asarray(enc_pos, dtype=cp.float32),
                    position_std,
                ).astype(cp.float32)
                log_joint_mark_intensity[0] = mligpu.estimate_log_joint_mark_intensity(
                    decoding_marks,
                    enc_marks,
                    mark_std,
                    interior_occupancy,
                    mean_rate,
                    position_distance=position_distance,
                )
                log_likelihood[:, is_track_interior] += np.nan_to_num(
                    log_joint_mark_intensity
                )

                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()

    log_likelihood[:, ~is_track_interior] = np.nan
    likelihood = scaled_likelihood(log_likelihood)
    return likelihood

LIKELIHOOD_FUNCTION = {
    "multiunit_likelihood": multiunit_likelihood,
    "spiking_likelihood_kde": spiking_likelihood_kde,
    "multiunit_likelihood_gpu": multiunit_likelihood_gpu,
    "multiunit_likelihood_integer_gpu": multiunit_likelihood_integer_gpu
}