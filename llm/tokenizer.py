#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 19:34:29 2026

@author: joaco
"""
import os
abs_path = os.path.abspath(__file__)
dname = os.path.dirname(abs_path)
os.chdir(dname)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
import re

preprocessed = re.split(r'(--|[.,:;¡!¿?()"\'_]|\s)', raw_text)
preprocessed = [s.strip() for s in preprocessed if s.strip()]
words = sorted(set(preprocessed))
words.extend(["<|unk|>", "<|eof|>"])
vocab = {x:i for i, x in enumerate(words)} 
 
class Tokenizer:
    def __init__(self, vocab:dict, pattern:str=r'(--|[.,:;?!()"\'_]|\s)' ):
        self.str2int = vocab
        self.int2str = {i:s for s,i in vocab.items()}
        self.pattern = pattern
    def encode(self, text:str):
        preprocessed = re.split(self.pattern, text)
        preprocessed = [s.strip() for s in preprocessed if s.strip()]
        preprocessed = [s if s in self.str2int else "<|unk|>" for s in preprocessed]
        ids = [self.str2int[s] for s in preprocessed]
        return ids
    def decode(self, ids:list):
        # Get string, each word separeted by whitespace
        text = " ".join([self.int2str[id] for id in ids])
        
        # Substitute whitespace's follew by symbol by only the symbol
        text = re.sub(r'\s+([.,?!"()\'])', r'\1', text)
        
        return text

print("first item in vocabulary: \n", list(vocab.items())[:10])
print("last items in vocabulary: \n", list(vocab.items())[1110:])

tokenizer = Tokenizer(vocab)
ids = tokenizer.encode("Hello i love C+, this is, a great day!")
tokenizer.decode(ids)