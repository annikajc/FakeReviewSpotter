import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import os
import numpy as np


class DatasetFakeReviews(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

        super().__init__()

    def __getitem__(self, idx):
        text, label = self.data.iloc[idx]['text_'], self.data.iloc[idx]['label']
        return text, label

    def __len__(self):
        return len(self.data)
    

class DataModuleFakeReviews(pl.LightningDataModule):
    def __init__(self, data="./data", batch_size=1, num_workers=0):
        super().__init__()
        # later step: NLP Augmentation
        
        # dataset/dataloader parameters
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers

        # empty datasets
        self.train = None
        self.val = None
        self.test = None
    
    def prepare_data_per_node(self):
        pass

    def setup(self, stage):
        # assign datasets for train, validation and test
        
        # load data in from csv
        reviews_data = pd.read_csv(os.path.join(self.data, "fake_reviews_dataset.csv"))
        reviews_data['label'] = np.where(reviews_data['label'] == 'CG', 1, 0)

        # split individual classes so dataset remains balanced
        trainval_1, test_1 = train_test_split(reviews_data[['text_', 'label']][reviews_data.label == 1], test_size=0.2, train_size=0.8, random_state=420)
        trainval_0, test_0 = train_test_split(reviews_data[['text_', 'label']][reviews_data.label == 0], test_size=0.2, train_size=0.8, random_state=420)

        train_1, val_1 = train_test_split(trainval_1, test_size=0.1, train_size=0.9, random_state=420)
        train_0, val_0 = train_test_split(trainval_0, test_size=0.1, train_size=0.9, random_state=420)

        # concatenate sets together and shuffle 
        comb_train = pd.concat([train_1, train_0])
        comb_val = pd.concat([val_1, val_0])
        comb_test = pd.concat([test_1, test_0])

        comb_train = shuffle(comb_train, random_state=420)
        comb_val = shuffle(comb_val, random_state=420)
        comb_test = shuffle(comb_test, random_state=420)

        # initialize datasets
        self.train = DatasetFakeReviews(data=comb_train.reset_index())
        self.val = DatasetFakeReviews(data=comb_val.reset_index())
        self.test = DatasetFakeReviews(data=comb_test.reset_index())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                                           pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                                           pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                                           pin_memory=True, persistent_workers=True)



