# -*- coding: utf-8 -*-

# Author Yasir Hussain (yaxirhuxxain@yahoo.com)

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


# "welcome to the tokenizers library."

print("*** Preparing Tokenizer ***")
tokenizer = Tokenizer(BPE(unk_token="<idf>"))
tokenizer.pre_tokenizer = Whitespace()

print("*** Preparing Trainer ***")
trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=1,
    show_progress=True,
    end_of_word_suffix="##",
    special_tokens=["<start>", "<sep>", "<end>", "<idf>", "<pad>", "<mask>"]
)

print("*** Reading Files ***")
java_path = "data/java"
csharp_path = "data/csharp"
java_files = [os.path.join(java_path, file) for file in os.listdir(java_path)]
csharp_files = [os.path.join(csharp_path, file) for file in os.listdir(csharp_path)]
train_files = java_files + csharp_files

print("*** Training Tokenizer ***")
tokenizer.train(train_files, trainer)

print("*** Saving Tokenizer ***")
tokenizer.save("data/tokenizer/code_tokenizer.json")

print("*** Done Tokenizer Training ***")

