import cv2
import pandas as pd
import torch
import parser
import argparse

from torch import nn
from tqdm.auto import tqdm
from model.decoder import DecoderWithAttention
from model.encoder import Encoder

tqdm.pandas()
from torch.utils.data import DataLoader, Dataset
from utils import *
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, Blur, RandomRotate90, CenterCrop
)
from albumentations.pytorch import ToTensorV2


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image


def get_transforms(*, data, args):
    if data == 'train':
        return Compose([
            Resize(height=args.image_size, width=args.image_size, p=1.0),
            RandomRotate90(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),

            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(args.image_size, args.image_size),
            CenterCrop(height=args.image_size, width=args.image_size, p=1.0),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER = init_logger(args.OUTPUT_DIR + 'train.log')
    test = pd.read_csv(args.test_path)
    tokenizer = torch.load(args.tokenizer_path)
    print(f"tokenizer.stoi: {tokenizer.stoi}")
    # return
    # 获得最长的单词长度
    data = pd.read_csv(args.data_path)
    args.max_length = data['SMILES_length'].max()
    print(args.max_length)
    seed_torch(seed=args.seed)

    smiles = test['SMILES']
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
            else:
                ch = x[i]
            i += 1
            text.append(ch)
        tmp = " ".join(str(i) for i in text)
        all_text.append(tmp)

    test["SMILES_text"] = all_text
    lengths = []
    tk0 = tqdm(test['SMILES_text'].values, total=len(test))
    for text in tk0:
        seq = tokenizer.text_to_sequence(text)
        length = len(seq) - 2
        lengths.append(length)
    test['SMILES_length'] = lengths
    print(test.head())

    test_labels = test['SMILES'].values
    test_dataset = TestDataset(test, transform=get_transforms(data='valid', args=args))

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    states = torch.load(f'{args.OUTPUT_DIR}{args.model_name}_tanimotoSimilarity_best.pth',
                        map_location=torch.device('cpu'))
    encoder = Encoder(args.model_name)
    encoder.load_state_dict(states['encoder'])
    encoder.to(device)

    decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                   embed_dim=args.embed_dim,
                                   decoder_dim=args.decoder_dim,
                                   vocab_size=len(tokenizer),
                                   dropout=args.dropout,
                                   device=device)
    decoder.load_state_dict(states['decoder'])
    decoder.to(device)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])

    encoder.eval()
    decoder.eval()
    # eval
    text_preds = valid_fn(args, test_loader, encoder, decoder, tokenizer, criterion, device)
    text_preds = [f"{text}" for text in text_preds]

    # scoring
    LevenshteinDistance = get_LevenshteinDistance(test_labels, text_preds)
    tanimotoSimilarity = get_tanimotoSimilarity(text_preds, test_labels)
    EM = exact_match_score(text_preds, test_labels)
    save_pred(test, args.OUTPUT_DIR + "SMILES_test_pred", text_preds, test_labels)

    LOGGER.info(f'test_LevenshteinDistance: {LevenshteinDistance:.4f}')
    LOGGER.info(f'test_tanimotoSimilarity: {tanimotoSimilarity}')
    LOGGER.info(f'test_EM: {EM}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=136)
    parser.add_argument('--print_freq', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='tf_efficientnet_b5')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--attention_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--decoder_dim', type=int, default=2048)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--OUTPUT_DIR', type=str, default='./HD_Dataset_synthetic_O_results/')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--test_path', type=str,
                        default='../../data_set/HNUCM_Hand_Drawn_Dataset/dataset/Heteroatom/Heteroatom_O_dataset.csv')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer/tokenizer.pth')
    parser.add_argument('--data_path', type=str, default='./data/CHO.csv')

    opt = parser.parse_args()
    main(opt)
