import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from typing import Any, List, Sequence, Tuple

class Actor(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self,
      num_actions: int,
      num_input: int,
      log_std_min: float = -20,
      log_std_max: float = 2):
    """Initialize."""
    super().__init__()
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max

    self.dense1 = layers.Dense(64, activation="relu")
    self.dense2 = layers.Dense(128, activation="relu")

    uniform_init = tf.keras.initializers.RandomUniform(-0.003, 0.003)
    self.log_std_layer = layers.Dense(num_actions, kernel_initializer=uniform_init, bias_initializer=uniform_init)
    self.mean_layer = layers.Dense(num_actions, kernel_initializer=uniform_init, bias_initializer=uniform_init)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.dense1(inputs)
    x = self.dense2(x)

    mu = tf.math.tanh(self.mean_layer(x))
    log_std = tf.math.tanh(self.log_std_layer(x))
    log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
    std = tf.math.exp(log_std)

    dist = tfp.distributions.Normal(mu, std)
    z = dist.sample()
    action = tf.math.tanh(z)

    log_prob = dist.log_prob(z) - tf.math.log(1 - tf.math.pow(action,2) + 1e-7)
    log_prob = tf.math.reduce_sum(log_prob,-1, keepdims=True)
    return action, log_prob

class CriticQ(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self,
      num_input: int,):
    """Initialize."""
    super().__init__()

    self.dense1 = layers.Dense(64, activation="relu")
    self.dense2 = layers.Dense(128, activation="relu")

    uniform_init = tf.keras.initializers.RandomUniform(-0.003, 0.003)
    self.critic = layers.Dense(1, kernel_initializer=uniform_init, bias_initializer=uniform_init)

  def call(self, state: tf.Tensor, action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    inputs = tf.concat((state, action), axis=-1)
    x = self.dense1(inputs)
    x = self.dense2(x)

    return self.critic(x)

class CriticV(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self,
       num_input: int,
    ):
    super().__init__()

    self.dense1 = layers.Dense(64, activation="relu")
    self.dense2 = layers.Dense(128, activation="relu")

    uniform_init = tf.keras.initializers.RandomUniform(-0.003, 0.003)
    self.critic = layers.Dense(1, kernel_initializer=uniform_init, bias_initializer=uniform_init)

  def call(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.dense1(state)
    x = self.dense2(x)

    return self.critic(x)

if __name__ == '__main__':
    model = ActorCritic(8, 27, 128)
    test = tf.random.normal((1,27))
    print(model(test))
