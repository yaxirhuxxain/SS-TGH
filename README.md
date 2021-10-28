# SS-TGH
This repository contains the dataset and code for the paper: **Boosting Source Code Suggestion with Self-Supervised Transformer Gated Highway**.

Please use the following citation if you use our dataset or code/implementation:
```
@article{hussain2021tgh,
    title={Boosting Source Code Suggestion with Self-Supervised Transformer Gated Highway},
    author={Yasir Hussain and Zhiqiu Huang and Yu Zhou},
    year={2021},
}
```

### General Info
- **data** folder contains the dataset and folds information.
- **encoder** folder contains the BPE & CUE encoder scripts.
- **github** folder contains the scripts to mine the projects from Github.
- **parser** folder contains scripts to parse the raw dataset.
- **tokenizer** folder contains a general-purpose tokenizer.
- **GatedHighway.py & TransformerGatedHighway.py** files contains model implementations.
- **SsLearner.py** file contains self-supervised learning methods.
- The rest of the scripts provide util functionalities.



### Models Info

The **TransformerGatedHighway.py** file contains the model implementation proposed in this paper. The **rnn.py** file contains model implementation for _codeLSTM_ baseline model which employes LSTM based recurrent neural network. We use the publicly available implementation of [BERT](https://huggingface.co/transformers/model_doc/bert.html) for classification as _CodeTran_ baseline. For _TravTrans_ baseline, we adopt the publically available implementation ([Link](https://github.com/facebookresearch/code-prediction-transformer/blob/main/model.py)) provided by the authors. The following hyperparameters are used for our implementation and the baselines:

- layers = 6
- heads = 6
- hidden = 300
- lr = 1e-3
- vocab = 30k

we refer the readers to the paper for more details.


