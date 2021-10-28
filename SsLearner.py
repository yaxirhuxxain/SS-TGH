# -*- coding: utf-8 -*-

# Author Yasir Hussain (yaxirhuxxain@yahoo.com)

def SliderMechanism(tokenized_datasets, context_size, n_workers=16):
        
        # Main data processing function that will concatenate all texts from our dataset and 
        # generate sequences with sliding window with context_size.
        def group_texts(examples):
            context_with_pred_size =  context_size+1 # +1 for next token pridiction
            
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

            result = {}
            for k, t in concatenated_examples.items():
                temp = []
                for i in range(0, len(t)-context_with_pred_size):
                    in_seq = t[i:i+context_with_pred_size]
                    if len(in_seq) !=context_with_pred_size:
                        raise f"Seq not in bound {in_seq}"
                    temp.append(in_seq)

                result[k] = temp

            return result

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=n_workers,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {context_size+1}",
        )
        
        return tokenized_datasets


def Chunk_iterator(tokenized_datasets, context_size,  n_workers=16):
        # Orignal scripts from:
        # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
        # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_plm.py

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of context_size.
        def group_texts(examples):
            context_with_pred_size =  context_size+1 # +1 for next token pridiction
            
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= context_with_pred_size+1:
                total_length = (total_length // context_with_pred_size) * context_with_pred_size

            # Split by chunks of max_len.
            result = {
                k: [t[i : i + context_with_pred_size] for i in range(0, total_length, context_with_pred_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=n_workers,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {context_size+1}",
        )

        return tokenized_datasets
