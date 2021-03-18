from torch.utils.data import Dataset


class ImagesAndFilepaths(Dataset):
    def __init__(self, img_dataset, filepaths):
        self.img_dataset = img_dataset
        self.filepaths = filepaths

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        img, target = self.img_dataset[idx]
        filepath = self.filepaths[idx]
        return img, target, filepath
