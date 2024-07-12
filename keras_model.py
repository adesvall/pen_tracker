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
from viz import plot_predictions
import pickle
import numpy as np

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
        print(input_shape)
        return input_shape[0], input_shape[1], 256

model = keras.models.Sequential(
    [
        keras.layers.InputLayer(shape=(None, 10)),

        myTCN(num_inputs=10,
                num_channels=[256] * 8,
                kernel_size=3,
                # dilations=(1,2, 1,2, 1,2, 1,2), # TODO
                dilations=None,
                dilation_reset=2,
                causal=False,
                dropout=0.2,
                use_norm="batch_norm",
        ),
        keras.layers.Dense(50),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(2)
    ]
)

def get_mask(yTrue):
    # zeros at the end of the sequence
    zeros = torch.flip(torch.norm(yTrue, dim=2) != 0, [1])
    # print(zeros)
    zeros = torch.cumsum(zeros, dim=1)
    # print(zeros)
    mask = torch.flip(zeros > 0, [1])
    # print(mask)
    return mask

# a = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 0], [0, 0, 0]], [[1, 2, 3], [0, 0, 0], [7, 8, 9], [0, 0, 0], [0, 0, 0]]], dtype=torch.float32)
# get_mask(a)


def my_loss(yTrue, yPred, gamma = 0.01, type=''):
    # print(yTrue.shape, yPred.shape)
    mask = get_mask(yTrue)
    # yTrue = torch.cumsum(yTrue, dim=1)
    # yPred = torch.cumsum(yPred, dim=1)
    yPred[~mask] = yTrue[~mask] # ie 0 for non cumsum

    dists = torch.cdist(yTrue, yPred)

    if type == 'frechet':
        return _SoftFrechet.apply(dists, gamma)
    elif type == 'dtw':
        return _SoftDTW.apply(dists, gamma)
    else:
        return torch.nn.MSELoss()(yTrue, yPred)


model.compile(optimizer='adam', loss=my_loss)
model.summary()

dataset = PenTrackerDataset()
print(len(dataset))
# train, test = random_split(dataset, [100, 35])
train = dataset
dataloader = DataLoader(train, batch_size=10, shuffle=True, collate_fn=dataset.collate_fn)

model.fit(dataloader, epochs=20, batch_size=50)
# pickle.dump(model, open('model.pkl', 'wb'))

ins, outs = next(iter(dataloader))
preds = model.predict(ins)

# préparation des données sorties

mask = get_mask(outs)

inputs = []
outputs = []
predictions = []

for i in range(len(mask)):
    length = mask[i].sum()
    inputs.append(ins[i, :length])
    outputs.append(np.cumsum(outs[i, :length], axis=0))
    predictions.append(np.cumsum(preds[i, :length], axis=0))


plot_predictions(inputs, outputs, predictions)

