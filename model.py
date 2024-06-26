import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from typing import Any, List, Sequence, Tuple


class ReinforcePolicy(tf.keras.Model):
    """Policy Network for a REINFORCE Algorithm"""

    def __init__(self,
                 num_actions: int,
                 num_input: int,
                 log_std_min: float = -3.5,
                 log_std_max: float = 2.5,
                 units_per_layer: int = 256,
                 action_scale: float = 1):
        """

        Parameters
        ----------
        num_actions
            Number of actions in the Environment
        num_input
            Number size of the Input of the network
        log_std_min
            Log of the minimum std of the normal distribution
        log_std_max
            Log of the maximum std of the normal distribution
        units_per_layer
            NUmber of perceptrons per dense Layer
        action_scale
            Scale applied to actions if they are not in range [-1,1]
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_scale = tf.constant(action_scale, dtype=tf.float32)

        self.dense1 = layers.Dense(units_per_layer, activation="relu")
        self.dense2 = layers.Dense(units_per_layer, activation="relu")

        uniform_init = tf.keras.initializers.RandomUniform(-0.1,
                                                           0.1,
                                                           seed=42)
        uniform_init = tf.keras.initializers.GlorotNormal(seed=42)
        self.log_std_layer = layers.Dense(num_actions,
                                          kernel_initializer=uniform_init,
                                          # kernel_initializer='glorot_uniform',
                                          bias_initializer='zeros')
        self.mean_layer = layers.Dense(num_actions,
                                       kernel_initializer=uniform_init,
                                       bias_initializer='zeros')

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Run the Network with the porvided Input

        Parameters
        ----------
        inputs
            Observation

        Returns
        -------
        Tuple

        """
        x = self.dense1(inputs)
        x = self.dense2(x)

        mu = tf.keras.activations.tanh(self.mean_layer(x))
        log_std = tf.keras.activations.tanh(self.log_std_layer(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max -
                                            self.log_std_min) * (log_std + 1)
        std = tf.math.exp(log_std)

        dist = tfp.distributions.Normal(mu, std, allow_nan_stats=False)
        z = dist.sample()
        action = tf.keras.activations.tanh(z)
        action = action * self.action_scale

        log_prob = dist.log_prob(z)
        log_prob = tf.math.reduce_sum(log_prob)
        return action, log_prob, mu, std, dist, tf.keras.activations.tanh(
            mu) * self.action_scale, z


class Actor(tf.keras.Model):
    """Actor for a SAC"""

    def __init__(self,
                 num_actions: int,
                 num_input: int,
                 log_std_min: float = -3.5,
                 log_std_max: float = 2.5,
                 units_per_layer: int = 256):
        """
        Parameters
        ----------
        num_actions
            Number of actions in the Environment
        num_input
            Number size of the Input of the network
        log_std_min
            Log of the minimum std of the normal distribution
        log_std_max
            Log of the maximum std of the normal distribution
        units_per_layer
            NUmber of perceptrons per dense Layer
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.dense1 = layers.Dense(units_per_layer, activation="relu")
        self.dense2 = layers.Dense(units_per_layer, activation="relu")

        uniform_init = tf.keras.initializers.RandomUniform(-0.01,
                                                           0.01,
                                                           seed=42)
        uniform_init = tf.keras.initializers.GlorotNormal(seed=42)
        self.log_std_layer = layers.Dense(num_actions,
                                          kernel_initializer=uniform_init,
                                          bias_initializer='zeros')
        self.mean_layer = layers.Dense(num_actions,
                                       kernel_initializer=uniform_init,
                                       bias_initializer='zeros')

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Run the Network with the porvided Input

        Parameters
        ----------
        inputs
            Observation

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, float, float]
            Tuple containing Predicted Action, log probability of the action,
            Mean and std of the sampled distirbution
        """
        x = self.dense1(inputs)
        x = self.dense2(x)

        mu = tf.keras.activations.tanh(self.mean_layer(x))
        log_std = tf.keras.activations.tanh(self.log_std_layer(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max -
                                            self.log_std_min) * (log_std + 1)
        std = tf.math.exp(log_std)

        dist = tfp.distributions.Normal(mu, std, allow_nan_stats=False)
        z = dist.sample()
        action = tf.keras.activations.tanh(z)

        log_prob = dist.log_prob(z) - tf.math.reduce_sum(
            tf.math.log(1 - tf.math.pow(action, 2) + 1e-7), -1, True)
        return action, log_prob, mu, std


class CriticQ(tf.keras.Model):
    """Q function approximator for SAC"""

    def __init__(self, num_input: int, units_per_layer: int = 256):
        """
        Parameters
        ----------
        num_input
            Dimension of the Network input
        units_per_layer
            number of units per dense layer
        """
        super().__init__()

        self.dense1 = layers.Dense(units_per_layer, activation="relu")
        self.dense2 = layers.Dense(units_per_layer, activation="relu")

        uniform_init = tf.keras.initializers.RandomUniform(-0.07,
                                                           0.07,
                                                           seed=41)
        self.critic = layers.Dense(1,
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer='zeros')

    def call(self, state: tf.Tensor,
             action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Call the network

        Parameters
        ----------
        state
            State of the environment
        action
            action to evaluate

        Returns
        -------
        tf.Tensor
            ouptut of the q function network
        """
        inputs = tf.concat((state, action), axis=-1)
        x = self.dense1(inputs)
        x = self.dense2(x)

        return self.critic(x)


class CriticV(tf.keras.Model):
    """Value Function approximator for SAC"""

    def __init__(self, num_input: int, units_per_layer: int = 256):
        """
        Parameters
        ----------
        num_input
            Dimension of the Network input
        units_per_layer
            number of units per dense layer
        """

        super().__init__()

        self.dense1 = layers.Dense(units_per_layer, activation="relu")
        self.dense2 = layers.Dense(units_per_layer, activation="relu")
        # self.dense3 = layers.Dense(units_per_layer, activation="relu")

        uniform_init = tf.keras.initializers.RandomUniform(-0.07,
                                                           0.07,
                                                           seed=40)
        self.critic = layers.Dense(1,
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer='zeros')

    def call(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Call the network

        Parameters
        ----------
        state
            State of the environment

        Returns
        -------
        tf.Tensor
            ouptut of the value function network
        """
        x = self.dense1(state)
        x = self.dense2(x)
        # x = self.dense3(x)

        return self.critic(x)


if __name__ == '__main__':
    model = ActorCritic(8, 27)
    test = tf.random.normal((1, 27))
    print(model(test))
