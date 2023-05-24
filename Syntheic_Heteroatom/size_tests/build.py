"""
This file reads a file of smiles strings and generates a 
hand-drawn chemical structure dataset of these molecules.

1. Collect smiles strings from txt file
2. Collect background images
3. For each smiles string:
    3a. Convert smiles string to ong of molecule
    3b. Augment molecule image using molecule 
        augmentation pipeline
    3c. Randomly select background image
    3d. Augment background image using background
        augmentation pipeline
    3e. Combine augmented molecule and augmented background
        using random weighted addition
    3f. Degrade total image
    3g. Save image to folder 
"""

import cv2
import os
import glob
import numpy as np
import random

from multiprocessing import Pool

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

from RDKit_modified.mol_drawing import MolDrawing
from RDKit_modified.local_canvas import Canvas
from RDKit_modified.mol_drawing import DrawingOptions
from degrade import degrade_img
from augment import augment_mol, augment_bkg

def get_smiles(filename):
    with open(filename) as f:
        lines = f.readlines()
    smiles = [s.split()[0] for s in lines]
    return smiles

def get_background_imgs(path):
    bkg_files = glob.glob("{}/*.png".format(path))
    bkgs = [cv2.imread(b) for b in bkg_files]
    return bkgs


def smiles_to_rdkitmod(s, i, img_dir):
    # 绘制化学分子结构式图像
    m = Chem.MolFromSmiles(s)
    AllChem.Compute2DCoords(m)
    # Chem.Kekulize可以将芳香苯环转换为Kekulize苯环
    m = Chem.Mol(m.ToBinary())
    Chem.Kekulize(m)
    # 使用修改源码的RDKit化学分子结构式绘制
    canvas = Canvas(size=(300, 300), name='{}/{}'.format(img_dir, i), imageType='svg')
    drawer = MolDrawing(canvas, drawingOptions=DrawingOptions)
    drawer.AddMol(m)
    canvas.flush()
    canvas.save()
    # 将SVG图像转为PNG图像
    svg = svg2rlg("{}/{}.svg".format(img_dir, i))
    renderPM.drawToFile(svg, "{}/{}.png".format(img_dir, i), fmt="PNG")
    os.remove("{}/{}.svg".format(img_dir, i))


def smiles_to_synthetic(s, i, img_dir):
    # 用RDKit先绘制图像
    smiles_to_rdkitmod(s, i, img_dir)
    mol = cv2.imread("{}/{}.png".format(img_dir, i))
    # 图像增强
    mol_aug = augment_mol(mol)
    # 随机添加背景
    if random.randint(1, 10) % 2 == 0:
        bkg = random.choice(bkgs)
        # 对背景增强
        bkg_aug = augment_bkg(bkg)
        # 将背景和分子图像结合
        p = np.random.uniform(0.1, 0.8)
        mol_bkg = cv2.addWeighted(bkg_aug, p, mol_aug, 1 - p, gamma=0)
        # 图像退化
        mol_bkg_deg = degrade_img(mol_bkg)
        # 保存图像
        cv2.imwrite("{}/{}.png".format(img_dir, i), mol_bkg_deg)
    else:
        # 图像退化
        mol_deg = degrade_img(mol_aug)
        # 保存图像
        cv2.imwrite("{}/{}.png".format(img_dir, i), mol_deg)


if __name__ == "__main__":

    # 修改RDKit源码，随机键长，角度
    Draw.DrawingOptions.wedgeBonds = False
    Draw.DrawingOptions.wedgeDashedBonds = False
    Draw.DrawingOptions.bondLineWidth = np.random.uniform(1, 7)

    path_bkg = "../backgrounds/"
    bkgs = get_background_imgs(path_bkg)

    stages = ['10w', '20w', '50w', '100w']
    for stage in stages:
        print("开始绘制{}张合成图像".format(stage))
        # 创建文件夹
        img_dir = "{}_images".format(stage)
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)

        # 得到SMILES编码
        smiles_file = "smiles_{}.txt".format(stage)
        smiles = get_smiles(smiles_file)

        for idx, s in enumerate(smiles):
            smiles_to_synthetic(s, idx, img_dir, stage)
    print("完成\n")
