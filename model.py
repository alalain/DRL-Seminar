import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from typing import Any, List, Sequence, Tuple


class Actor(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self,
                 num_actions: int,
                 num_input: int,
                 log_std_min: float = -4.5,
                 log_std_max: float = 2.5,
                 units_per_layer: int = 256):
        """Initialize."""
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.dense1 = layers.Dense(units_per_layer, activation="relu")
        self.dense2 = layers.Dense(units_per_layer, activation="relu")
        self.dense3 = layers.Dense(units_per_layer, activation="relu")

        # uniform_init = tf.keras.initializers.RandomUniform(-0.1,
        #                                                    0.1,
        #                                                    seed=42)
        uniform_init = tf.keras.initializers.GlorotNormal(seed=42)
        self.log_std_layer = layers.Dense(num_actions,
                                          kernel_initializer='glorot_uniform',
                                          bias_initializer='zeros')
        self.mean_layer = layers.Dense(num_actions,
                                       kernel_initializer='glorot_uniform',
                                       bias_initializer='zeros')

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)

        # mu = tf.math.tanh(self.mean_layer(x))
        # log_std = tf.math.tanh(self.log_std_layer(x))
        mu = tf.keras.activations.tanh(self.mean_layer(x))
        log_std = tf.keras.activations.tanh(self.log_std_layer(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max -
                                            self.log_std_min) * (log_std + 1)
        std = tf.math.exp(log_std)

        # dist = tfp.distributions.Normal(mu, std, allow_nan_stats=False)
        # dist = tfp.distributions.MultivariateNormalDiag(mu,
        #                                                 std,
        #                                                 allow_nan_stats=False)
        dist = tfp.distributions.Normal(mu, std, allow_nan_stats=False)
        z = dist.sample()
        action = tf.keras.activations.tanh(z)

        log_prob = dist.log_prob(z) - tf.math.reduce_sum(
            tf.math.log(1 - tf.math.pow(action, 2) + 1e-7), -1, True)
        #     log_prob = tf.math.reduce_sum(log_prob,-1, keepdims=True)
        return action, log_prob


class CriticQ(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_input: int, units_per_layer: int = 256):
        """Initialize."""
        super().__init__()

        self.dense1 = layers.Dense(units_per_layer, activation="relu")
        self.dense2 = layers.Dense(units_per_layer, activation="relu")
        self.dense3 = layers.Dense(units_per_layer, activation="relu")

        uniform_init = tf.keras.initializers.RandomUniform(-0.07,
                                                           0.07,
                                                           seed=41)
        self.critic = layers.Dense(1,
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer='zeros')

    def call(self, state: tf.Tensor,
             action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        inputs = tf.concat((state, action), axis=-1)
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)

        return self.critic(x)


class CriticV(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_input: int, units_per_layer: int = 256):
        super().__init__()

        self.dense1 = layers.Dense(units_per_layer, activation="relu")
        self.dense2 = layers.Dense(units_per_layer, activation="relu")
        self.dense3 = layers.Dense(units_per_layer, activation="relu")

        uniform_init = tf.keras.initializers.RandomUniform(-0.07,
                                                           0.07,
                                                           seed=40)
        self.critic = layers.Dense(1,
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer='zeros')

    def call(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)

        return self.critic(x)


if __name__ == '__main__':
    model = ActorCritic(8, 27)
    test = tf.random.normal((1, 27))
    print(model(test))
