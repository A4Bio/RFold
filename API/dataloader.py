from .dataset import RNADataset
from torch.utils.data import DataLoader


def load_data(data_name, batch_size, data_root, num_workers=8, **kwargs):
    if data_name == 'RNAStralign':
        test_set = RNADataset(path=data_root, dataname='test_600')
    elif data_name == 'ArchiveII':
        test_set = RNADataset(path=data_root, dataname='all_600')
    elif data_name == 'bpRNA':
        test_set = RNADataset(path=data_root, dataname='test')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    return test_loader