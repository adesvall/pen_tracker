import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from soft_dtw_cuda import _SoftDTW
from sleep_impl_frechet import _SoftFrechet
from _dataloader import PenTrackerDataset
from torch.utils.data import DataLoader, random_split

# sz = 100
# dataset_size = 1000
# inputs, outputs, theta2 = gen_folium_dataset(dataset_size, sz)
# print(inputs.shape, outputs.shape)

dataset = PenTrackerDataset()

train, test = random_split(dataset, [100, 35])
dataloader = DataLoader(train, batch_size=10, shuffle=True)


model = keras.models.Sequential(
    [
        keras.layers.InputLayer(shape=(None, 13)),
        keras.layers.Concatenate(axis=-1),
        
        # keras.layers.Conv1D(2, 3, padding ='same'),
        keras.layers.Dense(2),
        # keras.layers.Dense(50, activation='relu'),
        # keras.layers.Dense(100, activation='relu'),
    ]
)

def my_loss(x, y, gamma = 0.01):
    dists = torch.cdist(x, y)
    return _SoftFrechet.apply(dists, gamma)

model.compile(optimizer='adam', loss=my_loss)
model.summary()


model.fit(inputs, outputs, epochs=10, batch_size=10)
# pickle.dump(model, open('model.pkl', 'wb'))
preds = model.predict(inputs)


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

