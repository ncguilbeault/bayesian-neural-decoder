import pickle
import pandas as pd
import numpy as np
import track_linearization as tl
import replay_trajectory_classification as rtc
import os

class DataLoader:
    def __init__(self):
        super().__init__()
    
    @classmethod
    def load(cls, 
             results_path: str = "../../../datasets/decoder_data/clusterless_spike_decoding_results.pkl") -> dict:
        
        if not os.path.exists(results_path):
            raise Exception("Decoding results path incorrect.")

        with open(results_path, "rb") as f:
            results = pickle.load(f)
            decoding_results = results["decoding_results"]
            position_bins = decoding_results.position.to_numpy()[np.newaxis]
            predictions = decoding_results.acausal_posterior.to_numpy()
            position_data = results["linear_position"]
            spikes = results["spikes"]
            position_2D = results["position_2D"]
            position_bins_2D = results["position_bins_2D"][np.newaxis]

        return {
            "position_data": position_data,
            "spikes": spikes,
            "predictions": predictions,
            "position_bins": position_bins,
            "position_2D": position_2D,
            "position_bins_2D": position_bins_2D
        }
