import torch
from torch.utils.data import Dataset, DataLoader, random_split
# from gen_data import gen_folium_dataset

import pickle
import pandas as pd

# class FoliumDataset(Dataset):
#     def __init__(self, dataset_size, ts_size):
#         self.ts_size = ts_size
#         self.dataset_size = dataset_size
#         self.inputs, self.outputs, self.theta2 = gen_folium_dataset(dataset_size, ts_size)

#     def __len__(self):
#         return self.dataset_size
    
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.outputs[idx]


class PenTrackerDataset(Dataset):
    def __init__(self):
        input_data_file = "../KIHT_merge_SI_DEV/test_touch_sensor_arrs.pkl"
        with open(input_data_file, 'rb') as f:
            self.inputs = pickle.load(f)
            self.inputs.drop(columns=["mag_x", "mag_y", "mag_z"], inplace=True)

        output_truth_file = "../KIHT_merge_SI_DEV/test_touch_tablet_arrs.pkl"
        with open(output_truth_file, 'rb') as f:
            self.outputs = pickle.load(f)
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

    train, test = random_split(dataset, [100, 35])
    for input, output in train:
        print(input.shape, output.shape)

    dataloader = DataLoader(train, batch_size=10, shuffle=True)

    for i, (inputs, outputs) in enumerate(dataloader):
        print(inputs.shape, outputs.shape)

