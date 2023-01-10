import tensorflow_probability as tfp
import tensorflow as tf


tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.2, 0.2, 0.6]) # R, P, S

# distribution of next move
transition_distribution = tfd.Categorical(probs=[[0.2, 0.2, 0.6],
                                                 [0.6, 0.2, 0.2],
                                                 [0.2, 0.6, 0.2]])


observation_distribution = tfd.Multinomial(total_count=2., 
                                           probs=[[0.2, 0.6, 0.2],
                                                  [0.2, 0.2, 0.6],
                                                  [0.6, 0.2, 0.2]])

model = tfd.HiddenMarkovModel(
            initial_distribution = initial_distribution,
            transition_distribution = transition_distribution,
            observation_distribution = observation_distribution,
            num_steps = 7)

# notimplemented error: mean is not implemented in hiddenmarkov model
mean = model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())

