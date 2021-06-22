# state data: 100 signals of 50 length
python3 data_generator/simulations_threshold_spikes.py
python3 evaluation/baselines.py --data 'simulation_spike' --explainer 'fit' --train


# b FITExplainer.attribute