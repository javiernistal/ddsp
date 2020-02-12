import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import gin

tfkl = tf.keras.layers

@gin.register
class Generator(tfkl.Layer):
    def __init__(self,
               output_splits=(('amps', 1), ('harmonic_distribution', 40)),
               ch=512,
               name='generator'):
        super().__init__(name=name)
        self.output_splits = output_splits
        self.n_out = sum([v[1] for v in output_splits])

        self.layers = Sequential()
        self.layers.add(Dense(ch, input_dim=128))
        self.layers.add(tf.keras.layers.LeakyReLU(0.2))

        self.layers.add(Dense(ch))
        self.layers.add(tf.keras.layers.LeakyReLU(0.2))

        self.layers.add(Dense(ch))
        self.layers.add(tf.keras.layers.LeakyReLU(0.2))

        # self.layers.add(Dense(img_rows*img_cols*channels, activation='tanh'))
        
        # self.layers.compile(loss='binary_crossentropy', optimizer=optimizer)        
    
    def call(self, conditioning):
        """Updates conditioning with dictionary of decoder outputs."""
        x = self.generate(conditioning)
        outputs = nn.split_to_dict(x, self.output_splits)

        if isinstance(outputs, dict):
          conditioning.update(outputs)
        else:
          raise ValueError('Decoder must output a dictionary of signals.')
        return conditioning

    def generate(self, conditioning):
        f, l, z = (conditioning['f0_scaled'],
                   conditioning['ld_scaled'],
                   conditioning['z'])

        # Initial processing.
        f = self.f_stack(f)
        l = self.l_stack(l)
        z = self.z_stack(z)

        # Run an RNN over the latents.
        x = tf.concat([f, l, z], axis=-1) if self.append_f0_loudness else z
        x = self.layers(x)
        x = tf.concat([f, l, x], axis=-1)

        # Final processing.
        x = self.out_stack(x)
        return self.dense_out(x)
