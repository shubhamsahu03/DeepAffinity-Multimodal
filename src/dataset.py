import torch
from torch_geometric.data import InMemoryDataset
from .processor import process_parallel

class BindingDBDataset(InMemoryDataset):
    def __init__(self, root, df=None, embedding_dict=None, split_name='train', transform=None):
        self.df = df
        self.embedding_dict = embedding_dict
        self.split_name = split_name
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def processed_file_names(self):
        return [f'bindingdb_{self.split_name}.pt']

    def process(self):
        print(f"Processing {self.split_name} set...")
        data_list = process_parallel(self.df, self.embedding_dict)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
