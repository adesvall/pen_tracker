import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from soft_dtw_cuda import _SoftDTW
from sleep_impl_frechet import _SoftFrechet
from _dataloader import PenTrackerDataset
from torch.utils.data import DataLoader, random_split
from pytorch_tcn import TCN
from keras.src.utils.torch_utils import TorchModuleWrapper

# sz = 100
# dataset_size = 1000
# inputs, outputs, theta2 = gen_folium_dataset(dataset_size, sz)
# print(inputs.shape, outputs.shape)


# doc TCN
# https://github.com/paul-krug/pytorch-tcn


class myTCN(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.tcn = TCN(**kwargs)
    
    def call(self, x):
        x = x.transpose(1, 2)
        return self.tcn.call(x).transpose(1, 2)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], None, 256

model = keras.models.Sequential(
    [
        keras.layers.InputLayer(shape=(None, 10)),

        myTCN(num_inputs=10,
                num_channels=[256] * 4,
                kernel_size=3,
                dilations=None, # TODO
                causal=False,
                dropout=0.2,
                use_norm="batch_norm",
        ),
        keras.layers.Dense(50),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(2)
    ]
)

def my_loss(x, y, gamma = 0.01, type='dtw'):
    # print(x.shape, y.shape)
    dists = torch.cdist(x, y)
    if type == 'frechet':
        return _SoftFrechet.apply(dists, gamma)
    elif type == 'dtw':
        return _SoftDTW.apply(dists, gamma)


model.compile(optimizer='adam', loss=my_loss)
model.summary()


dataset = PenTrackerDataset()
train, test = random_split(dataset, [100, 35])
dataloader = DataLoader(train, batch_size=10, shuffle=True, collate_fn=dataset.collate_fn)

model.fit(dataloader, epochs=10, batch_size=10)
# pickle.dump(model, open('model.pkl', 'wb'))
# preds = model.predict(inputs)


# import matplotlib.pyplot as plt

# plt.title('Input, Output, Prediction')
# k = 4
# for i in range(k):
#     plt.subplot(2, k, i + 1)
#     plt.gca().set_title(f"Input {i}")
#     plt.gca().set_aspect('equal')
#     plot_trajectory(inputs[i], plt.gca())
#     plt.subplot(2, k, k + i + 1)
#     plt.gca().set_aspect('equal')
#     plot_trajectory(outputs[i], plt.gca())
#     plot_trajectory(preds[i], plt.gca(), ['black'])
# plt.show()

