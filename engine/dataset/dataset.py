from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """"""

    def __init__(self):
        super(ImageDataset, self).__init__()
        """step of load your data from files"""
        # TODO
        ...

    def __getitem__(self, index):
        """define how to get img-target pair from index"""
        # TODO
        ...

    def __len__(self):
        # TODO
        ...

    def evaluation(self):
        """calculate evaluation metrics"""
        # TODO
        ...

    def __repr__(self):
        """print dataset information"""
        # TODO
        ...
