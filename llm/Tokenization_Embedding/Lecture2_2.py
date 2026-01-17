#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 07:10:12 2026

@author: joaco
"""
import tiktoken
import os 
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
#tiktoken.encode("hello! how are you?")
tokenizer = tiktoken.get_encoding("gpt2")
ids = tokenizer.encode("Akwirw ier")
strings = tokenizer.decode(ids)

print("ids: ", ids)
for id in ids:
    print(id, ": ",tokenizer.decode([id]))
print("strings: ", strings)