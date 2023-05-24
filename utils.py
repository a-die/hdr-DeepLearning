# -- coding: utf-8 --
# @Time : 2022/11/4 10:03
# @Author : 欧阳亨杰
# @File : utils.py
# ====================================================
# Helper functions
# ====================================================
import codecs
import csv
import math
import os
import time

import Levenshtein
import numpy as np
import torch
from rdkit import Chem, DataStructs
import random
import xlwt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts


from torch.nn.utils.rnn import pack_padded_sequence


class Tokenizer(object):

    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(args,train_loader, encoder, decoder, criterion,
             encoder_optimizer, decoder_optimizer, epoch,
             encoder_scheduler, decoder_scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # 改为训练模式
    encoder.train()
    decoder.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels, label_lengths) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        batch_size = images.size(0)
        # 通过编码器提取特征
        features = encoder(images)
        # 特征输入解码器
        predictions, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, labels, label_lengths)
        targets = caps_sorted[:, 1:]
        predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        loss = criterion(predictions, targets)
        losses.update(loss.item(), batch_size)
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
        decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            encoder_optimizer.step()
            decoder_optimizer.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()
        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Encoder Grad: {encoder_grad_norm:.4f}  '
                  'Decoder Grad: {decoder_grad_norm:.4f}  '
            .format(
                epoch + 1, step, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,
                remain=timeSince(start, float(step + 1) / len(train_loader)),
                encoder_grad_norm=encoder_grad_norm,
                decoder_grad_norm=decoder_grad_norm,
            ))
    return losses.avg


def valid_fn(args,valid_loader, encoder, decoder, tokenizer, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # 改为验证模式
    encoder.eval()
    decoder.eval()
    text_preds = []
    start = end = time.time()
    for step, (images) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        with torch.no_grad():
            # 通过编码器提取特征
            features = encoder(images)
            # 特征输入解码器
            predictions = decoder.predict(features, args.max_length, tokenizer)
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        _text_preds = tokenizer.predict_captions(predicted_sequence)
        text_preds.append(_text_preds)
        batch_time.update(time.time() - end)
        end = time.time()
        if step % args.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
            .format(
                step, len(valid_loader), batch_time=batch_time,
                data_time=data_time,
                remain=timeSince(start, float(step + 1) / len(valid_loader)),
            ))
    text_preds = np.concatenate(text_preds)
    return text_preds


# ====================================================
# Utils
# ====================================================
def get_LevenshteinDistance(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score


def get_tanimotoSimilarity(pred, labels):
    all = 0
    for i in range(len(pred)):
        ref1 = Chem.MolFromSmiles(pred[i])
        ref2 = Chem.MolFromSmiles(labels[i])
        try:
            fp1 = Chem.RDKFingerprint(ref1)
            fp2 = Chem.RDKFingerprint(ref2)
            Tan = DataStructs.TanimotoSimilarity(fp1, fp2)
            all += Tan
        except Exception as e:
            pass
    return all / len(pred)


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ====================================================
# scheduler
# ====================================================
def get_scheduler(args,optimizer):
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, verbose=True,
                                      eps=args.eps)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr, last_epoch=-1)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=1, eta_min=args.min_lr, last_epoch=-1)
    return scheduler


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


def exact_match_score(pred, labels):
    exact_match = 0
    for ref, hypo in zip(pred, labels):
        if np.array_equal(ref, hypo):
            exact_match += 1
    return exact_match / float(max(len(pred), 1))


def save_pred(data, name, preds, labels):
    workbook = xlwt.Workbook(encoding='utf-8')  # 设置一个workbook，其编码是utf-8
    worksheet = workbook.add_sheet("test_sheet")  # 新增一个sheet
    id1 = list(data['IDs'])
    pred = preds  # 列1
    labels = list(labels)  # 列2
    worksheet.write(0, 0, label='id')
    worksheet.write(0, 1, label='pred')  # 将‘列1’作为标题
    worksheet.write(0, 2, label='labels')  # 将‘列2’作为标题
    for i in range(len(pred)):
        # 循环将a和b列表的数据插入至excel
        worksheet.write(i + 1, 0, label=id1[i])
        worksheet.write(i + 1, 1, label=pred[i])
        worksheet.write(i + 1, 2, label=labels[i])
    workbook.save(name + ".xls")  # 这里save需要特别注意，文件格式只能是xls，不能是xlsx，不然会报错

# Import smtplib for the actual sending function
import smtplib
from email.mime.text import MIMEText


def send_email(subject="No subject", content="I am boring"):
    mail_host = "smtp.qq.com"
    mail_user = "oyhj1019@qq.com"
    mail_pw = "iznjyfwzxdasbeib"  # 授权码
    sender = "oyhj1019@qq.com"
    receiver = "oyhj1019@qq.com"

    # Create the container (outer) email message.
    msg = MIMEText(content, "plain", "utf-8")
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    try:
        smtp = smtplib.SMTP_SSL(mail_host, 465)  # 实例化smtp服务器
        smtp.login(mail_user, mail_pw)  # 登录
        smtp.sendmail(sender, receiver, msg.as_string())
        print("Email send successfully")
    except smtplib.SMTPException:
        print("Error: email send failed")
