# neural-decoding

This package provides tools for decoding neural data using a bayesian state-space model. It is designed to implement the code from [here](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification) and can be used for running online decoding in bonsai.

You can read more about the technique here: 

Eric L Denovellis, Anna K Gillespie, Michael E Coulter, Marielena Sosa, Jason E Chung, Uri T Eden, Loren M Frank (2021). Hippocampal replay of experience at real-world speeds. eLife 10:e64505.

# Installation

To install, run the following commands:

```
cd /path/to/bayesian-neural-decoder
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

>[!NOTE] The package uses `cupy-cuda12x` for GPU support which requires CUDA v12 to be installed seperately.