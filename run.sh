# state data: 10 signals of 24 length
python3 data_generator/simulations_threshold_spikes.py
python3 -m evaluation.baselines --data 'simulation_spike' --explainer 'fit' --train


