Estimate the Fisher information matrix of a simulated forward model using a
conditional invertible neural network (cINN) that approximates the
observation likelihood p(x | z).

### Pipeline:
    1. Generate synthetic (z, x) pairs from a prior and forward
       model (data.py: SimulationData, priors, forward models).
    2. Train a cINN approximating p(x | z) (train.Trainer), or load a
       previously trained checkpoint, see ReadME.md and yaml config file.
    3. Compare the Fisher information estimated from the flow's score
       function (eval.FisherEstimator) against the analytic
       Fisher information for the chosen forward model over a grid. 
