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

# ----------------------------------------------------------------------

## Configuration options

For data generation, need to specify a prior and a forward transformation model.

Below shows how to specify them in config.yaml

### Options for prior: gaussian, uniform. 
  prior:
    type: uniform
    low: [-2.0, -2.0]
    high: [2.0, 2.0]

  prior:
    type: gaussian
    mu: [0.0, 0.0]
    std: [3.0, 3.0]

### Options for forward_model: linear, polynomial, coupled_polynomial, spiral, banana, tanh
  forward_model:
    type: linear
    coupling: 0.7

  forward_model:
    type: polynomial

  forward_model:
    type: coupled_polynomial

  forward_model:
    type: banana
    curvature: 0.5

  forward_model:
    type: spiral
    turns: 1.5

  For tanh, choose width a bit larger than the prior's support so the prior
  mostly covers the responsive (non-saturated) part of the tanh, rather
  than its flat tails where the Fisher information drops toward zero
  everywhere. 

  forward_model:
    type: tanh
    scale: 1.0
    width: 2.0
