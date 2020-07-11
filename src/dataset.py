import pandas as pd
import config
from torch.utils.data import Dataset
import numpy as np
import os
import torch

class FakeNewsDataset(Dataset):

    def __init__(self,train=True):
        num_rows = 20000
        self.train=train
        if self.train == True:
            self.data = pd.read_csv(config.TRAIN_DATA_PATH,usecols=config.FEATURES).head(num_rows).values
            self.labels =  pd.read_csv(config.TRAIN_DATA_PATH,usecols=config.TARGET).head(num_rows).values
        else:
            self.data = pd.read_csv(config.TEST_DATA_PATH,usecols=config.FEATURES).iloc[65000:].values
        self.n_samples = len(self.data)
        self.tokenizer = config.TOKENIZER
    
    def _truncate_tokens(self,tokens,l1):

        l = config.MAX_LEN //2
        if l1 > l:
            return np.array(tokens[:l-1] + [102] + tokens[-l:])
        else:
            return np.array(tokens[:l] + tokens[-l:])
    
    def _get_tokenized_data(self,text,add_special_tokens=True):
        return self.tokenizer.encode(text,add_special_tokens=add_special_tokens)
    
    def _padding(self,tokenized_text):
        max_len = config.MAX_LEN
        if len(tokenized_text) > max_len:
            return self._truncate_tokens(tokenized_text,config.MAX_LEN //2 -1)
        return np.array(tokenized_text + [0] * (max_len - len(tokenized_text)))

    def _get_attention_mask(self,padded_array):
        return np.where(padded_array != 0,1,0)
    
    def _get_segment_ids(self, l1,l2):
        if l1 + l2 > config.MAX_LEN:
            segment_ids = [0] * (config.MAX_LEN//2) + [1] * (config.MAX_LEN//2)
        else:
            segment_ids = [0] * l1 + [1] * l2 + [0] * (config.MAX_LEN - (l1 + l2))
        return  np.array(segment_ids)
    
    def _preprocessing_data(self,data): 

        tokenized_text_a = self._get_tokenized_data(data[0])
        tokenized_text_b = self._get_tokenized_data(data[1],add_special_tokens=False)
        tokenized_text_b.append(102)
        tokenized_text = tokenized_text_a + tokenized_text_b

        l1,l2 = len(tokenized_text_a),len(tokenized_text_b)
        if l1 + l2 > config.MAX_LEN:
            padded_array = self._truncate_tokens(tokenized_text,l1)
        else:
            padded_array = self._padding(tokenized_text)
        segment_ids = self._get_segment_ids(l1,l2)
        attention_mask = self._get_attention_mask(padded_array)
        return padded_array, attention_mask, segment_ids

    def __getitem__(self, index):

        padded_array, attention_mask, segment_ids =  self._preprocessing_data(self.data[index])
        if self.train == True:
            labels =  self.labels[index][0]
            res = {'input_ids': padded_array, 'segment_ids': segment_ids, 'attention_mask': attention_mask, 'labels': labels}
        else:
            res = {'input_ids': padded_array, 'segment_ids': segment_ids, 'attention_mask': attention_mask}
        return res

    def __len__(self):  
        return self.n_samples
