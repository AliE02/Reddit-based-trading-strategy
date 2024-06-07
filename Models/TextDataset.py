from typing import List
import torch
from torch.utils.data import Dataset, pad_sequence


class TextDataset(Dataset):
    def __init__(self, X: List[str], y: List[int]=None):
        """
        Initializes the TextDataset.

        Args:
        X: List[str]: A list of texts.
        y: List[int]: A list of labels corresponding to the texts.
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]
    
def collate_batch(batch):
    """
    Collates a batch of data.

    Args:
    batch: List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples of tensors.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors.
    """
    X, y = zip(*batch)
    X = [torch.tensor(x) for x in X]
    X = pad_sequence(X, batch_first=True)
    y = torch.tensor(y)
    return X, y
