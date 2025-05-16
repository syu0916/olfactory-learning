import tensorflow as tf
import keras_tuner as kt
import hyperparameters as hyper
from keras import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input
from preprocess import get_data


class GenomeMLP(kt.HyperModel):

    def build(self, hp):

        input1 = Input(shape=(hyper.num_features,), name="input1")

        u = hp.Int(f"units-1", min_value = 1024, max_value = 2048, step = 1024)
        x = Dense(units = 1024, activation = "relu") (input1)

        u = hp.Int(f"units-2", min_value = 512, max_value = 1024, step = 512)
        x = Dense(units = 768, activation = "relu") (x)

        x = Dropout(0.25)(x)

        u = hp.Int(f"units-3", min_value = 512, max_value = 1024, step = 128)
        x = Dense(units = 512, activation = "relu")(x)

        x = Dropout(0.25)(x)

        u = hp.Int(f"units-4", min_value = 128, max_value = 256, step = 128)
        x = Dense(units = 128, activation = "relu")(x)


        # for i in range(4):
        #     self.model.add(
        #         Dense(
        #             units=hp.Int(f"units{i}", min_value=128, max_value=1024, step=128),
        #             activation="relu",
        #         )
        #     )
        #     if i == 1 or i == 2:
        #         self.model.add(
        #             Dropout(hp.Float(f"dropout_rate{i}", 0.10, 0.50, step = 0.10))
        #         )
        #         self.model.add(BatchNormalization())
        

        output = Dense(hyper.classes, activation="softmax")(x)

        # model = Model(inputs=[input1, input2], outputs=output)
        model = Model(inputs=input1, outputs=output)

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
    
