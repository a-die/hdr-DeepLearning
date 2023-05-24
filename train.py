# -- coding: utf-8 --
# @Time : 2022/11/4 10:00
# @Author : 欧阳亨杰
# @File : train.py
import argparse
from tqdm.auto import tqdm

tqdm.pandas()
import pandas as pd
from preprocess import Preprocess
import cv2
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from utils import *
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, Blur
)
from albumentations.pytorch import ToTensorV2
from model.decoder import DecoderWithAttention
from model.encoder import Encoder
import warnings

warnings.filterwarnings('ignore')


def bms_collate(batch):
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)


def get_transforms(*, data, size):
    if data == 'train':
        return Compose([
            Resize(size, size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(size, size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


class TrainDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df['file_path'].values
        self.labels = df['SMILES_text'].values
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
        label = self.labels[idx]
        label = self.tokenizer.text_to_sequence(label)
        label_length = len(label)
        label_length = torch.LongTensor([label_length])
        return image, torch.LongTensor(label), label_length


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


# ====================================================
# Train loop 直接训练SMILES
# ====================================================
def train_loop(args, LOGGER, tokenizer, train, val):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"==========training ==========")
    # ====================================================
    # loader
    # ====================================================
    valid_labels = val['SMILES'].values

    train_dataset = TrainDataset(train, tokenizer, transform=get_transforms(data='train', size=args.size))
    valid_dataset = TestDataset(val, transform=get_transforms(data='valid', size=args.size))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=bms_collate)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    encoder = Encoder(args.model_name, pretrained=True)
    encoder.to(device)
    encoder_optimizer = Adam(encoder.parameters(), lr=args.encoder_lr, weight_decay=args.weight_decay, amsgrad=False)
    encoder_scheduler = get_scheduler(args, encoder_optimizer)

    decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                   embed_dim=args.embed_dim,
                                   decoder_dim=args.decoder_dim,
                                   vocab_size=len(tokenizer),
                                   dropout=args.dropout,
                                   device=device)
    decoder.to(device)
    decoder_optimizer = Adam(decoder.parameters(), lr=args.decoder_lr, weight_decay=args.weight_decay, amsgrad=False)
    decoder_scheduler = get_scheduler(args, decoder_optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])

    best_LevenshteinDistance = np.inf
    best_em = 0
    best_tanimotoSimilarity = 0

    best_LevenshteinDistance_preds = []
    best_tanimotoSimilarity_preds = []
    best_em_preds = []

    lossData = [[]]
    LevenshteinDistanceData = [[]]
    tanimotoSimilarityData = [[]]
    EMData = [[]]
    iter = 0

    for epoch in range(args.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(args, train_loader, encoder, decoder, criterion,
                            encoder_optimizer, decoder_optimizer, epoch,
                            encoder_scheduler, decoder_scheduler, device)

        # eval
        text_preds = valid_fn(args, valid_loader, encoder, decoder, tokenizer, criterion, device)
        text_preds = [f"{text}" for text in text_preds]

        # scoring
        LevenshteinDistance = get_LevenshteinDistance(valid_labels, text_preds)
        tanimotoSimilarity = get_tanimotoSimilarity(text_preds, valid_labels)
        EM = exact_match_score(valid_labels, text_preds)

        LOGGER.info(f"labels: {valid_labels[:5]}")
        LOGGER.info(f"preds: {text_preds[:5]}")

        lossData.append([iter, avg_loss])  # 先转成普通tensor，再转成numpy形式
        LevenshteinDistanceData.append([iter, LevenshteinDistance])
        tanimotoSimilarityData.append([iter, tanimotoSimilarity])
        EMData.append([iter, EM])
        iter += 1

        if isinstance(encoder_scheduler, ReduceLROnPlateau):
            encoder_scheduler.step(LevenshteinDistance)
        elif isinstance(encoder_scheduler, CosineAnnealingLR):
            encoder_scheduler.step()
        elif isinstance(encoder_scheduler, CosineAnnealingWarmRestarts):
            encoder_scheduler.step()

        if isinstance(decoder_scheduler, ReduceLROnPlateau):
            decoder_scheduler.step(LevenshteinDistance)
        elif isinstance(decoder_scheduler, CosineAnnealingLR):
            decoder_scheduler.step()
        elif isinstance(decoder_scheduler, CosineAnnealingWarmRestarts):
            decoder_scheduler.step()

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Levenshtein Distance: {LevenshteinDistance:.4f}')
        LOGGER.info(f'Epoch {epoch + 1} - EM: {EM:.4f}')
        LOGGER.info(f'Epoch {epoch + 1} - TanimotoSimilarity: {tanimotoSimilarity:.4f}')

        if tanimotoSimilarity > best_tanimotoSimilarity:
            best_tanimotoSimilarity = tanimotoSimilarity
            best_tanimotoSimilarity_preds = text_preds
            LOGGER.info(f'Epoch {epoch + 1} - Save Best TanimotoSimilarity: {best_tanimotoSimilarity:.4f} Model')
            torch.save({'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        },
                       args.OUTPUT_DIR + f'{args.model_name}_tanimotoSimilarity_best.pth')

        if LevenshteinDistance < best_LevenshteinDistance:
            best_LevenshteinDistance = LevenshteinDistance
            best_LevenshteinDistance_preds = text_preds
            LOGGER.info(f'Epoch {epoch + 1} - Save Best LevenshteinDistance: {best_LevenshteinDistance:.4f} Model')
            torch.save({'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        },
                       args.OUTPUT_DIR + f'{args.model_name}_LevenshteinDistanc_best.pth')
        if EM > best_em:
            best_em = EM
            best_em_preds = text_preds
            LOGGER.info(f'Epoch {epoch + 1} - Save Best EM: {best_em:.4f} Model')
            torch.save({'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        },
                       args.OUTPUT_DIR + f'{args.model_name}_em_best.pth')
    data_write_csv(args.OUTPUT_DIR + "direct_loss.csv", lossData)
    data_write_csv(args.OUTPUT_DIR + "direct_LevenshteinDistance.csv", LevenshteinDistanceData)
    data_write_csv(args.OUTPUT_DIR + "direct_tanimotoSimilarity.csv", tanimotoSimilarityData)
    data_write_csv(args.OUTPUT_DIR + "direct_EM.csv", EMData)

    save_pred(val, args.OUTPUT_DIR + "direct_SMILES_pred", best_LevenshteinDistance_preds, valid_labels)
    save_pred(val, args.OUTPUT_DIR + "direct_tanimotoSimilarity_preds", best_tanimotoSimilarity_preds, valid_labels)
    save_pred(val, args.OUTPUT_DIR + "direct_em_preds", best_em_preds, valid_labels)


def main(args):
    if os.path.exists(args.OUTPUT_DIR) is False:
        os.makedirs(args.OUTPUT_DIR)
    train = pd.read_csv(args.train_path)
    val = pd.read_csv(args.val_path)
    preprocess = Preprocess(train, val)
    global tokenizer
    data, tokenizer, train, val = preprocess.get_preprocess_pd()
    args.max_length = data['SMILES_length'].max()
    print(args)
    LOGGER = init_logger(args.OUTPUT_DIR + 'train.log')
    seed_torch(seed=args.seed)
    start_time = time.time()
    train_loop(args, LOGGER, tokenizer, train, val)
    elapsed = time.time() - start_time
    content = f'训练完成,耗时{elapsed:.0f}s'
    send_email(subject="Training finished", content=content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_freq', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--train_path',type=str, default='../../Syntheic_Heteroatom/Syntheic_Heteroatom_train.csv')
    parser.add_argument('--val_path', type=str, default='../../Syntheic_Heteroatom/Syntheic_Heteroatom_train.csv')
    parser.add_argument('--scheduler', type=str,
                        default='CosineAnnealingLR')  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--factor', type=float, default=0.2)  # ReduceLROnPlateau
    parser.add_argument('--patience', type=int, default=4)  # ReduceLROnPlateau
    parser.add_argument('--eps', type=float, default=1e-6)  # ReduceLROnPlateau
    parser.add_argument('--T_max', type=int, default=4)  # CosineAnnealingLR
    parser.add_argument('--T_0', type=int, default=4)  # CosineAnnealingWarmRestarts
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_lr', type=float, default=4e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=int, default=5)
    parser.add_argument('--attention_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--decoder_dim', type=int, default=2048)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--OUTPUT_DIR', type=str, default='./HD_Dataset_results/')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
