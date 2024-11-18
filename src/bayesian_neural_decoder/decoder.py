import replay_trajectory_classification as rtc
from replay_trajectory_classification.core import scaled_likelihood, get_centers
from replay_trajectory_classification.likelihoods import _SORTED_SPIKES_ALGORITHMS, _ClUSTERLESS_ALGORITHMS
from replay_trajectory_classification.likelihoods.spiking_likelihood_kde import combined_likelihood, poisson_log_likelihood
from replay_trajectory_classification.likelihoods.multiunit_likelihood import estimate_position_distance, estimate_log_joint_mark_intensity

from .likelihood import LIKELIHOOD_FUNCTION

import numpy as np
import pickle as pkl
import cupy as cp

class Decoder():
    def __init__(self):
        super().__init__()

    def decode(self, data: np.ndarray):
        raise NotImplementedError
    
    @classmethod
    def load(cls, filename: str):
        with open(filename, "rb") as f:
            return cls(pkl.load(f))

class ClusterlessSpikeDecoder(Decoder):
    def __init__(self, model_dict: dict):
        self.decoder = model_dict["decoder"]

        encoding_model = self.decoder.encoding_model_
        self.encoding_marks = encoding_model["encoding_marks"]
        self.mark_std = encoding_model["mark_std"]
        self.encoding_positions = encoding_model["encoding_positions"]
        self.position_std = encoding_model["position_std"]
        self.occupancy = encoding_model["occupancy"]
        self.mean_rates = encoding_model["mean_rates"]
        self.summed_ground_process_intensity = encoding_model["summed_ground_process_intensity"]
        self.block_size = encoding_model["block_size"]
        self.bin_diffusion_distances = encoding_model["bin_diffusion_distances"]
        self.edges = encoding_model["edges"]

        self.place_bin_centers = self.decoder.environment.place_bin_centers_
        self.place_bin_centers_1D = self.place_bin_centers.squeeze()
        self.is_track_interior = self.decoder.environment.is_track_interior_.ravel(order="F")
        self.st_interior_ind = np.ix_(self.is_track_interior, self.is_track_interior)

        self.likelihood_function = LIKELIHOOD_FUNCTION[self.decoder.clusterless_algorithm]

        if "gpu" in self.decoder.clusterless_algorithm:
            self.is_track_interior_gpu = cp.asarray(self.is_track_interior)
            self.occupancy = cp.asarray(self.occupancy)
            self.interior_place_bin_centers = cp.asarray(
                self.place_bin_centers[self.is_track_interior], dtype=cp.float32
            )
            self.interior_occupancy = cp.asarray(
                self.occupancy[self.is_track_interior_gpu], dtype=cp.float32
            )

        else:
            self.is_track_interior_gpu = None
            self.interior_place_bin_centers = np.asarray(
                self.place_bin_centers[self.is_track_interior], dtype=np.float32
            )
            self.interior_occupancy = np.asarray(
                self.occupancy[self.is_track_interior], dtype=np.float32
            )

        self.n_position_bins = self.is_track_interior.shape[0]
        self.n_track_bins = self.is_track_interior.sum()

        self.initial_conditions = self.decoder.initial_conditions_[self.is_track_interior].astype(float)
        self.state_transition = self.decoder.state_transition_[self.st_interior_ind].astype(float)

        self.posterior = None
        super().__init__()

    @classmethod
    def load(cls, filename: str = "../../../datasets/decoder_data/clusterless_spike_decoder.pkl"):
        return super().load(filename)
    
    def decode(self,
               data: np.ndarray):

        likelihood = self.likelihood_function(
            data, 
            self.summed_ground_process_intensity, 
            self.encoding_marks, 
            self.encoding_positions, 
            self.mean_rates, 
            self.is_track_interior, 
            self.interior_place_bin_centers, 
            self.position_std, 
            self.mark_std, 
            self.interior_occupancy, 
            self.n_track_bins
        )

        if self.posterior is None:
            self.posterior = np.full((self.n_position_bins), np.nan, dtype=float)
            self.posterior[self.is_track_interior] = self.initial_conditions * likelihood[0, self.is_track_interior]

        else:
            self.posterior[self.is_track_interior] = self.state_transition.T @ self.posterior[self.is_track_interior] * likelihood[0, self.is_track_interior]

        norm = np.nansum(self.posterior)
        self.posterior /= norm

        return (self.posterior, self.place_bin_centers_1D)

class SortedSpikeDecoder(Decoder):
    def __init__(self, model_dict: dict):
        self.decoder = model_dict["decoder"]

        self.is_track_interior = self.decoder.environment.is_track_interior_.ravel(order="F")
        self.st_interior_ind = np.ix_(self.is_track_interior, self.is_track_interior)
        self.n_position_bins = self.is_track_interior.shape[0]

        self.initial_conditions = self.decoder.initial_conditions_[self.is_track_interior].astype(float)
        self.state_transition = self.decoder.state_transition_[self.st_interior_ind].astype(float)
        self.place_fields = np.asarray(self.decoder.place_fields_)
        self.place_bin_centers = self.decoder.environment.place_bin_centers_.squeeze()
        self.conditional_intensity = np.clip(self.place_fields, a_min=1e-15, a_max=None)

        self.likelihood_function = LIKELIHOOD_FUNCTION[self.decoder.sorted_spikes_algorithm]

        self.posterior = None
        super().__init__()

    @classmethod
    def load(cls, filename: str = "../../../datasets/decoder_data/sorted_spike_decoder.pkl"):
        return super().load(filename)

    def decode(
            self,
            data: np.ndarray
        ):

        likelihood = self.likelihood_function(data, self.conditional_intensity, self.is_track_interior)

        if self.posterior is None:
            self.posterior = np.full((self.n_position_bins), np.nan, dtype=float)
            self.posterior[self.is_track_interior] = self.initial_conditions * likelihood[0]

        else:
            self.posterior[self.is_track_interior] = self.state_transition.T @ self.posterior[self.is_track_interior] * likelihood[0]

        norm = np.nansum(self.posterior)
        self.posterior /= norm

        return (self.posterior, self.place_bin_centers)