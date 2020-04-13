import torch.utils.data
import torch


class FilteredDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, wanted_labels):
        self.parent = dataset
        indices = []
        for index, (img, lab) in enumerate(dataset):
            if isinstance(wanted_labels, list):
                if lab in wanted_labels:
                    indices.append(index)
            elif lab == wanted_labels:
                indices.append(index)        
        self.indices = indices

    def __getitem__(self, index):
        return self.parent[self.indices[index]]

    def __len__(self):
        return len(self.indices)
