import random
import numpy as np
import pickle
from num2words import num2words
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch 

class KLLoss(torch.nn.Module):
    def __init__(self, error_metric=torch.nn.KLDivLoss(reduction='batchmean')):
        super().__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label):
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2)
        return loss
    
    
def gen_mask(batch_TLs):
    batch_sets = [set([tuple(item) for item in sublist]) for sublist in batch_TLs]
    
    unique_TLs = [random.choice(sublist) for sublist in batch_TLs ]
    num_unique = len(unique_TLs)
    matrix = np.zeros((len(batch_TLs),num_unique), dtype=np.float32)
    
    TL_to_index = {}
    for j, TL in enumerate(unique_TLs):
        if TL in TL_to_index:
            TL_to_index[TL].append(j)
        else:
            TL_to_index[TL] = [j]
    for i, TL_set in enumerate(batch_sets):
        for TL in TL_set:
            if TL in TL_to_index:
                for j in TL_to_index[TL]:
                    matrix[i, j] = 1.0
    
    count = np.sum(matrix)
    return unique_TLs,matrix, count


class U3TDataset(Dataset):
    def __init__(self, data):
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        obs = self.data[index][0]
        act = self.data[index][1]
        TLs = self.data[index][2] #template language
        length = self.data[index][3]
        NLs = self.data[index][4] #trajectory level natural language
        return obs,act,TLs,length,NLs
    
def split_dataset(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    x_train,x_test = train_test_split(data , train_size=0.8)
    return U3TDataset(x_train),U3TDataset(x_test)

    


