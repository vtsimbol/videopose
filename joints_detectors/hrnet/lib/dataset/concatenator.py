from torch.utils.data import Dataset


class Concatenator(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        self._lens = [len(ds) for ds in self._datasets]

    def __getitem__(self, index):
        ds_i, local_i = self._parse_index(index)
        return self._datasets[ds_i][local_i]

    def __len__(self):
        return sum(self._lens)

    def _parse_index(self, index):
        ds_i = 0
        for l in self._lens:
            if index >= l:
                ds_i += 1
                index -= l
            else:
                break
        return ds_i, index
