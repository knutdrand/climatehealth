import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


def create_model(temperature, rainfall, dengue):
    temperature_effect = tfp.sts.LinearRegression(
          design_matrix=tf.reshape(temperature - np.mean(temperature),
                                   (-1, 1)), name='temperature_effect')
    week_effect = tfp.sts.Seasonal(
        num_seasons=52,
        observed_time_series=dengue,
        name='week_effect')
    residual_level = tfp.sts.Autoregressive(
        order=1,
        observed_time_series=dengue, name='residual')
    model = tfp.sts.Sum([temperature_effect,
                         week_effect,
                         residual_level],
                        observed_time_series=dengue)
    
