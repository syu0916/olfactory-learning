import tensorflow as tf
import keras_tuner as kt
import hyperparameters as hyper
from keras import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Layer
from preprocess import get_data
import numpy as np


class GenomeMLP(kt.HyperModel):

    def build(self, hp):
        # Define two inputs
        input1 = Input(shape=(hyper.num_features,), name="input1")  # Replace input1_dim with actual dimension
        input2 = Input(shape=(hyper.num_cell_types,), name="input2")  # Replace input2_dim with actual dimension

        x = input1

        u = hp.Int(f"one-hot", min_value = 512, max_value = 2048, step = 9)
        one_hots = Dense(units = u, activation = "relu")(input2)
        x = Concatenate(name = "concatenate_i2")([x, one_hots])

        u = hp.Int(f"units-1", min_value = 1024, max_value = 2048, step = 1024)
        x = Dense(units = u, activation = "relu") (x)

        u = hp.Int(f"units-2", min_value = 512, max_value = 1024, step = 512)
        x = Dense(units = u, activation = "relu") (x)

       

        u = hp.Int(f"units-3", min_value = 512, max_value = 1024, step = 128)
        x = Dense(units = u, activation = "relu")(x)
        

        # class MaxScaler(Layer):
        #     def call(self, inputs):
        #         max_val = tf.reduce_max(inputs, axis=1, keepdims=True)
        #         return max_val * hp.Int(f"scalar-max", min_value = 2, max_value = 10, step = 2)


        u = hp.Int(f"units-4", min_value = 128, max_value = 256, step = 128)
        x = Dense(units = u, activation = "relu")(x)
        


        output = Dense(hyper.classes, activation="softmax")(x)

        # model = Model(inputs=[input1, input2], outputs=output)
        model = Model(inputs=[input1, input2], outputs=output)

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            # Tune whether to shuffle the data in each epoch.
            epochs = hp.Int("epochs", 10, 50, step=50),
            batch_size=hp.Int('batch_size', 32, 256, step=32),
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )
    
