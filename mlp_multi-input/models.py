import tensorflow as tf
import hyperparameters as hp
from keras.layers import Dropout, Dense, BatchNormalization, Model, Concatenate
from keras import Sequential

def train(model, train_data, test_data):
    weight_folder = "weights/best_weights.h5"

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=weight_folder,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_sparse_categorical_accuracy',  # Monitor validation accuracy
        verbose=1,
        mode='max'  # 'max' because we want to maximize accuracy
    )

    model.fit(
        x = train_data, 
        validation_data = test_data, 
        epochs = hp.epochs, 
        batch_size = hp.batch_size,
        callbacks=[callback]
        )

def test(model, test_data):
    model.evaluate(
        x = test_data, 
        verbose = 1
    )


class GenomeMLP(tf.keras.Model):
    def __init__(self):
        super(GenomeMLP, self).__init__()
        self.model = Sequential(
            [
                Dense(units=l1, activation="relu", name="dense"),

                Dense(units=l2, activation="relu", name="dense_1"),
                BatchNormalization(name="batch_normalization"),
                Dropout(d1, name="dropout"),

                Dense(units=l3, activation="relu", name="dense_2"),
                BatchNormalization(name="batch_normalization_1"),
                Dropout(d2, name="dropout_1"),

                Dense(units=l4, activation="relu", name="dense_3"),
                Dense(units=hp.classes, activation="softmax", name="dense_4")
            ]
        )



        # self.model.add(tf.keras.layers.InputLayer(input_shape=(hp.num_features,)))
        # self.model.add(Dense(units=l1, activation="relu", name="dense"))
        # # Create an input layer to ensure the model has a valid input tensor

        # self.model.add(Dense(units=l2, activation="relu", name="dense_1"))
        # self.model.add(BatchNormalization(name="batch_normalization"))
        # self.model.add(Dropout(d1, name="dropout"))

        # self.model.add(Dense(units=l3, activation="relu", name="dense_2"))
        # self.model.add(BatchNormalization(name="batch_normalization_1"))
        # self.model.add(Dropout(d2, name="dropout_1"))

        # self.model.add(Dense(units=l4, activation="relu", name="dense_3"))
        # self.model.add(Dense(units=hp.classes, activation="softmax", name="dense_4"))

    def call(self, x):
        # Forward pass through each layer
        return self.model(x)

