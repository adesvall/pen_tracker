import torch
from torch.utils.data import Dataset, DataLoader, random_split
# from gen_data import gen_folium_dataset
import pickle

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
        input_data = "../KIHT_merge_SI_DEV/test_touch_sensor_arrs.pkl"
        with open(input_data, 'rb') as f:
            self.inputs = pickle.load(f)

        output_truth = "../KIHT_merge_SI_DEV/test_touch_tablet_arrs.pkl"
        with open(output_truth, 'rb') as f:
            self.outputs = pickle.load(f)
        
        for c in self.outputs:
            print(c.columns)
        for c in self.inputs:
            print(c.columns)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

if __name__ == '__main__':
    dataset = PenTrackerDataset()
    print(len(dataset))
    train, test = random_split(dataset, [100, 35])
    dataloader = DataLoader(train, batch_size=10, shuffle=True)

    for i, (inputs, outputs) in enumerate(dataloader):
        print(inputs.shape, outputs.shape)

