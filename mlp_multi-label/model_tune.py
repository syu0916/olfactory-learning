import tensorflow as tf
import keras_tuner as kt
import hyperparameters as hyper
from keras import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Layer
from preprocess import get_data
import numpy as np

"""
    The file used to build our model. GenomeMLP inherits kt.HyperModel, which is 
    used to hypertune in the main.py script. 
"""

class GenomeMLP(kt.HyperModel):

    def build(self, hp):
        # Define two inputs
        input1 = Input(shape=(hyper.num_features,), name="input1")  # Replace input1_dim with actual dimension


        x = input1

        # u = hp.Int(f"one-hot", min_value = 512, max_value = 2048, step = 9)
        # one_hots = Dense(units = u, activation = "relu")(input2)
        # x = Concatenate(name = "concatenate_i2")([x, one_hots])
        
        u = hp.Int(f"units-1", min_value = 1024, max_value = 2048, step = 256)
        x = Dense(units = u, activation = "relu") (x)

        x = BatchNormalization()(x)

        u = hp.Int(f"units-2", min_value = 512, max_value = 1024, step = 512)
        x = Dense(units = u, activation = "relu") (x)

        x = BatchNormalization()(x)

        # x = Dropout(0.25, name="dropout1")(x)
        u = hp.Int(f"units-3", min_value = 512, max_value = 1024, step = 256)
        x = Dense(units = u, activation = "relu")(x)

        # x = Dropout(0.1, name="dropout2")(x)
        x = BatchNormalization()(x)
        
        u = hp.Int(f"units-4", min_value = 128, max_value = 256, step = 128)
        x = Dense(units = u, activation = "relu")(x)
        
        condition = Dense(hyper.classes, activation="softmax", name = "condition-out")(x)
        ct = Dense(hyper.num_cell_types, activation = "softmax", name = "cell-type-out")(x)


        # model = Model(inputs=[input1, input2], outputs=output)
        model = Model(inputs=[input1], outputs=[condition, ct])

        model.compile(
            optimizer='adam',
            loss={
                'condition-out': 'sparse_categorical_crossentropy',
                'cell-type-out': 'sparse_categorical_crossentropy',
            },
            loss_weights={
                'condition-out': 0.90,
                'cell-type-out': 0.10,  
            },
            metrics={
                'condition-out': ['accuracy'],
                'cell-type-out': ['accuracy']
            }
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
    
