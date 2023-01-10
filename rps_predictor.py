import tensorflow_probability as tfp
import tensorflow as tf


tfd = tfp.distributions
initial_distributions = tfd.Categorical(probs=[0.333, 0.333, 0.333]) # R, P, S
transition_distribution = tfd.Categorical(probs=[[0.2, 0.2, 0.6],
                                                 [0.6, 0.2, 0.2],
                                                 [0.2, 0.6, 0.2]])

# TODO: figure out observation distribution
# prev play distribution based on state (current move)
observation_distribution = tfd.Normal(loc=[0, 0, 0],
                                      scale=[0, 0, 0])
