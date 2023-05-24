# -- coding: utf-8 --
# @Time : 2023/2/5 11:53
# @Author : 欧阳亨杰
# @File : preprocess.py

from tqdm.auto import tqdm
import pandas as pd
import torch

from utils import Tokenizer

tqdm.pandas()


class Preprocess(object):
    def __init__(self, train, val):
        self.train = train
        self.val = val

    def get_preprocess_pd(self):
        smiles = self.train['SMILES']
        all_text = list()
        for x in smiles:
            i = 0
            text = list()
            while i < len(x):
                if x[i] == '[':
                    ch = ""
                    while x[i] != ']':
                        ch += x[i]
                        i += 1
                    ch += x[i]
                elif x[i] == 'B' and x[i + 1] == 'r':
                    ch = 'Br'
                    i += 1
                elif x[i] == 'C' and i + 1 < len(x) and x[i + 1] == 'l':
                    ch = 'Cl'
                    i += 1
                else:
                    ch = x[i]
                i += 1
                text.append(ch)
            tmp = " ".join(str(i) for i in text)
            all_text.append(tmp)

        self.train["SMILES_text"] = all_text

        smiles = self.val['SMILES']
        all_text = list()

        for x in smiles:
            i = 0
            text = list()
            while i < len(x):
                if x[i] == '[':
                    ch = ""
                    while x[i] != ']':
                        ch += x[i]
                        i += 1
                    ch += x[i]
                elif x[i] == 'B' and x[i + 1] == 'r':
                    ch = 'Br'
                    i += 1
                elif x[i] == 'C' and i + 1 < len(x) and x[i + 1] == 'l':
                    ch = 'Cl'
                    i += 1
                else:
                    ch = x[i]
                i += 1
                text.append(ch)
            tmp = " ".join(str(i) for i in text)
            all_text.append(tmp)
        self.val["SMILES_text"] = all_text
        data = pd.concat([self.train, self.val], axis=0)

        # ====================================================
        # create tokenizer
        # ====================================================
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data['SMILES_text'].values)
        lengths = []
        tk0 = tqdm(data['SMILES_text'].values, total=len(data))
        for idx,text in enumerate(tk0):
            seq = tokenizer.text_to_sequence(text)
            length = len(seq) - 2
            lengths.append(length)
        data['SMILES_length'] = lengths
        print(len(tokenizer), max(lengths))
        print(f'data.shape: {data.shape}')
        print(data.head())

        return data, tokenizer, self.train, self.val


if __name__ == '__main__':
    train = pd.read_csv('../../data_set/Syntheic_Heteroatom/pipeline_stages/rdkitp_train.csv')
    val = pd.read_csv('../../data_set/Syntheic_Heteroatom/pipeline_stages/rdkitp_val.csv')
    preprocess = Preprocess(train, val)
    data, tokenizer, train, val = preprocess.get_preprocess_pd()
    data.to_csv('data/pipline_stages.csv')
    torch.save(tokenizer, 'tokenizer/tokenizer_pipline_stages.pth')
