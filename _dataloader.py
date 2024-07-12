import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise, Convolve, Resize, Pool, Dropout
# from gen_data import gen_folium_dataset
from viz import plot_trajectory
import matplotlib.pyplot as plt

import pickle
import pandas as pd
import numpy as np
import random

# def minimal_data_gen(in_ds, out_ds, batch_size=16, paddingvalue=0., maskvalue=-50, aug=False, win_size=5):
#     """
#     Minimal data generator for time series data.
#     :param in_ds: list of input time series (dataframes)
#     :param out_ds: list of output time series (dataframes)
#     :param batch_size: number of samples in a batch
#     :param paddingvalue: value for padding shorter time series (default=0.)
#     :param maskvalue: value for masking shorter output time series (default=-50)
#     :param aug: enable data augmentation (default=False)
#     :param win_size: window size for framing (default=5)
#     :return: generator yielding batches of (input, output) arrays
#     """
#     def tsFraming(data, win_size, channels):
#         framed_data = np.zeros((data.shape[0], data.shape[1] - 2 * win_size, win_size * 2 + 1, channels))
#         for i in range(data.shape[1] - 2 * win_size):
#             framed_data[:, i] = data[:, i:i + 2 * win_size + 1]
#         return framed_data
    
#     feature_list_in = in_ds[0].columns.values
#     feature_list_out = out_ds[0].columns.values
    
#     while True:
#         for c in range(0, len(in_ds)-batch_size, batch_size):
#             TIME_STEPS = max(len(di) for di in in_ds[c:c + batch_size] if len(di) > 0)
#             in_ = np.zeros((batch_size, TIME_STEPS, len(feature_list_in))).astype('float')
#             out_ = np.zeros((batch_size, TIME_STEPS, len(feature_list_out))).astype('float')
            
#             for i in range(c, c + batch_size):
                
#                 train_x = in_ds[i] if len(in_ds[i]) == TIME_STEPS else pd.concat([in_ds[i], pd.DataFrame(paddingvalue, index=np.arange(int(TIME_STEPS - len(in_ds[i]))), columns=feature_list_in)])
#                 train_y = out_ds[i] if len(out_ds[i]) == TIME_STEPS else pd.concat([out_ds[i], pd.DataFrame(maskvalue, index=np.arange(int(TIME_STEPS - len(out_ds[i]))), columns=feature_list_out)])
                
#                 in_[i - c] = train_x.to_numpy()
#                 out_[i - c] = train_y.to_numpy()
            
#             if aug:
#                 my_augmenter = Crop(size=int(TIME_STEPS // random.randrange(2, 8, 1)))
#                 in_, out_ = my_augmenter.augment(in_, out_)
            
#             # in_ = tsFraming(in_, win_size, len(feature_list_in))
#             yield in_, out_

def collate_fn(batch):
    max_len = max(len(x[0]) for x in batch)

    padded_in = [torch.nn.functional.pad(xin, (0, 0, 0, max_len - len(xin))) for xin, xout in batch]
    padded_out = [torch.nn.functional.pad(xout, (0, 0, 0, max_len - len(xout))) for xin, xout in batch]
        
    padded_in = torch.stack(padded_in)
    padded_out = torch.stack(padded_out)

    return padded_in, padded_out

class PenTrackerDataset(Dataset):
    def __init__(self):
        self.collate_fn = collate_fn

        input_data_file = "data/train_touch_sensor_arrs.pkl"
        output_truth_file = "data/train_touch_tablet_arrs.pkl"

        with open(input_data_file, 'rb') as f:
            self.inputs = pickle.load(f)
            for df in self.inputs:
                df.drop(columns=["mag_x", "mag_y", "mag_z"], inplace=True)
        with open(output_truth_file, 'rb') as f:
            self.outputs = pickle.load(f)
            for df in self.outputs:
                df.drop(columns=['imu-millis', 'millis', 'pres'], inplace=True)
        # self.outputs[0].to_csv("output.csv")
        
        print("Inputs columns :", self.inputs[0].columns)
        print("Output columns :", self.outputs[0].columns)

        self.inputs = [torch.tensor(df.values) for df in self.inputs]
        self.outputs = [torch.tensor(df.values) for df in self.outputs]

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


if __name__ == '__main__':
    dataset = PenTrackerDataset()
    print(len(dataset))

    # train, test = random_split(dataset, [100, 32])
    # for input, output in train:
    #     print(input.shape, output.shape)

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=dataset.collate_fn)

    for i, (inputs, outputs) in enumerate(dataloader):
        
        print(inputs.shape, outputs.shape)
        o = outputs[0]
        o = np.cumsum(o, axis=0)
        plot_trajectory(o, plt.gca())
        plt.show()
        if i == 0:
            break

