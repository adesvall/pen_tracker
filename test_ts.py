import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization
import keras
from tcn import TCN




raw_input = Input(shape=(None, 13))
# x = tf.keras.layers.Concatenate(axis=-1)([raw_input[:, :, :9], raw_input[:, :, 12]])
x = TCN(nb_filters=256,
                    kernel_size=3,
                    nb_stacks=4,
                    dilations=(1, 2),
                    padding='same',
                    return_sequences=True,
                    dropout_rate=0.2,
                    use_batch_norm=True,
                    )(raw_input)
x = Dense(50, activation='linear')(x)
x = BatchNormalization()(x)
out_put = Dense(2)(x)

m = keras.models.Model(inputs=[raw_input], outputs=[out_put])
m.summary()

