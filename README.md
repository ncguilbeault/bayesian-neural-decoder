# neural-decoding

This package provides tools for decoding neural data using a bayesian state-space model.

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

For GPU support, also run:

```
pip install cupy
```