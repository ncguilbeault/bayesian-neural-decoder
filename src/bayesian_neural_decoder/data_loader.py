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
             dataset_path: str = "../../../datasets/decoder_data", 
             results_filename: str = "clusterless_spike_decoding_results.pkl") -> dict:
        
        decoding_results_filename = os.path.join(dataset_path, results_filename)
        if not os.path.exists(decoding_results_filename):
            raise Exception("Dataset incorrect. Missing decoding results file.")

        with open(decoding_results_filename, "rb") as f:
            results = pickle.load(f)
            decoding_results = results["decoding_results"]
            position_bins = decoding_results.position.to_numpy()[np.newaxis]
            predictions = decoding_results.acausal_posterior.to_numpy()
            position_data = results["linear_position"]
            model_inputs = results["model_inputs"]

        return {
            "position_data": position_data,
            "model_inputs": model_inputs,
            "decoding_results": predictions,
            "position_bins": position_bins
        }
