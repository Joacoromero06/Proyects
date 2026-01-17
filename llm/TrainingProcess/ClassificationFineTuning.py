#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 09:30:03 2026

@author: joaco
"""

import urllib.request
import zipfile
import os
from pathlib import Path
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
def download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")       
        return
    # Downloads the file
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extracted_path)
    
    # Unzips the file
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")
    
if __name__ == "__main__":
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
# %%
import pandas as pd


def create_balanced_dataset(df):
    # Counts the instances  of “spam”
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample ham instances to match the number of “spam” instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    
    # Combines ham subset with “spam”
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df


def splitting_dataset(df, train_fracc, val_fracc):
    # Random shuffle dataframe
    df = df.sample(
        frac=1, random_state=123
        ).reset_index(drop=True)
    # Splitting
    n = len(df)
    n_train, n_val = int(n*train_fracc), int(n*val_fracc)
    train_df = df[:n_train]
    val_df = df[n_train: n_train + n_val]
    test_df = df[n_train + n_val:]
    return train_df, val_df, test_df
# %% 
if __name__ == "__main__":
    # get data fram from dataset
    df = pd.read_csv(data_file_path, sep='\t', names=["Label", "Text"])
    
    # count apparition of labels classes
    df["Label"].value_counts()
    
    # balanced dataset
    balanced_df = create_balanced_dataset(df)
    
    # Change vocabulary of classes
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    
    # Split data set
    train_df, val_df, test_df = splitting_dataset(balanced_df, 
                                                  train_fracc=0.7, 
                                                  val_fracc=0.1)
    
    # Save splitted sets
    train_df.to_csv("train.csv", index=None)
    val_df.to_csv("val.csv", index=None)
    test_df.to_csv("test.csv", index=None)

# %%
from torch.utils.data import Dataset
from torch import tensor, long
import tiktoken 

class SpamDataset(Dataset):
    """
    Load text from CSV train file
    Tokenizes using tiktoken
    Allow to truncate messages to 'max_length'. Usually model limit
    Padds messages to 'max_length' using 'pad_tok'. Need for batches
    """
    def __init__(self, csv, T, max_length=None, pad_tok=50256):
        self.data = pd.read_csv(csv)
        self.encode_texts = [ T.encode(s) for s in self.data["Text"]]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            # Truncates sentences if they are longer
            self.max_length = max_length
            self.encode_texts = [ ids[:max_length] for ids in self.encode_texts ]
            
        # Pad ids to longest length
        self.encode_texts = [ 
ids + [pad_tok] * (self.max_length - len(ids)) for ids in self.encode_texts]
    def __getitem__(self, index):
        ids = self.encode_texts[index]
        label = self.data.iloc[index]["Label"]
        return tensor(ids, dtype=long), tensor(label, dtype=long)
    def __len__(self):
        return len(self.data)
    def _longest_encoded_length(self):
        may = 0
        for ids in self.encode_texts:
            len_i = len(ids)
            if may < len_i:
                may = len_i
        return may
        
# %%

T = tiktoken.get_encoding("gpt2")

# Annote the  token id of padding token
print(T.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# Create Dataset
train = SpamDataset(csv="train.csv" , T=T, max_length=None)
val = SpamDataset(csv="val.csv" , T=T, max_length=None)
test = SpamDataset(csv="test.csv" , T=T, max_length=None)

# Compare context_limit of model with the max_length sequence of Dataset
print(train.max_length)

# Creating Data Loader
from torch.utils.data import DataLoader        
train_loader = DataLoader(dataset=train,
                          shuffle=True,
                          batch_size=8,
                          num_workers=0,
                          drop_last=True)
val_loader = DataLoader(dataset=val,
                          shuffle=True,
                          batch_size=8,
                          num_workers=0,
                          drop_last=False)
test_loader = DataLoader(dataset=test,
                          shuffle=True,
                          batch_size=8,
                          num_workers=0,
                          drop_last=False)
for X, Y in train_loader:
    print(X.shape, Y.shape)
    break
for X, Y in val_loader:
    print(X.shape, Y.shape)