import os

from torch.utils.data import Dataset


class LabeledDirDataset(Dataset):
    def __init__(self, root_folder: str, data_key: str = "x", label_key: str = "y"):
        self.root_folder = root_folder
        self.data_key = data_key
        self.label_key = label_key
        self.class_map = {folder_name: label for (label, folder_name) in enumerate(os.listdir(self.root_folder))}
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for folder_name in self.class_map:
            files_list = os.listdir(os.path.join(self.root_folder, folder_name))
            for file in files_list:
                data.append((os.path.join(self.root_folder, folder_name, file), self.class_map[folder_name]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data, label = self.data[index]
        return {self.data_key: data, self.label_key: label}
