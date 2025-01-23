import os
import chess
import chess.pgn
import time
import math
from tqdm import tqdm
import random
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

piece_to_index = {
    'P': 0,  'N': 1,  'B': 2,  'R': 3,  'Q': 4,  'K': 5,  # White pieces
    'p': 6,  'n': 7,  'b': 8,  'r': 9,  'q': 10, 'k': 11,  # Black pieces
}
class ChessDataset(Dataset):
    def __init__(self, pgn_file, device='cpu'):
        self.device = device
        self.moves_data = []
        
        # 读取PGN文件中的游戏并生成每一步的p, q, r
        with open(pgn_file) as f:
            while game:=chess.pgn.read_game(f):
                moves = list(game.mainline_moves())
                board = game.board()
                for move in moves[:-1]:
                    flip = (board.turn==chess.WHITE)
                    p = self.board2vec(board, flip=(not flip))
                    legal_moves = list(board.legal_moves)
                    pseudo_move = random.choice(legal_moves)
                    board.push(pseudo_move)
                    r = self.board2vec(board, flip=flip)
                    board.pop()
                    board.push(move)
                    q = self.board2vec(board, flip=flip)
                    self.moves_data.append((p, q, r))

    def __len__(self):
        return len(self.moves_data)
    
    def board2vec(self, board, flip=False):
        vec = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_index = piece_to_index[piece.symbol()]
                row, col = divmod(square, 8)
                if flip:
                    # 翻转行
                    row = 7 - row
                    # 翻转棋子颜色
                    if piece_index < 6:  
                        piece_index += 6 
                    else: 
                        piece_index -= 6  
                vec[piece_index, row, col] = 1
        vec = torch.tensor(vec, dtype=torch.float32).to(self.device)
        return vec
        
    def __getitem__(self, idx):
        return self.moves_data[idx]

class ChessValueNetwork(nn.Module):
    def __init__(self):
        super(ChessValueNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层1
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 卷积层2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 卷积层3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 最大池化层
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 2048),  # 输入特征的大小将取决于卷积层的输出
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1)
        )

    def forward(self, x):
        x = self.conv(x)  # 经过卷积层
        x = x.view(x.size(0), -1)  # 展平成 [Batch Size, Features]
        x = self.fc(x)  # 经过全连接层
        return x

def objective_function(model, p, q, r, kappa=10.0):
    # Forward pass: Compute scores for p, q, and r
    f_p = model(p).squeeze()  # Score for p
    f_q = model(q).squeeze() # Score for q
    f_r = model(r).squeeze() # Score for r
    # Loss components
    # Loss A: Ensure f(q) > f(r) (optimal move vs random move)
    loss_a = -torch.log(F.sigmoid(f_r - f_q)).mean()
    # Loss B: Ensure f(p) + f(q) close to zero (soft equality constraint)
    loss_b = -torch.log(F.sigmoid(kappa * (f_p + f_q))).mean()
    # Loss C: Ensure -f(p) - f(q) close to zero (soft equality constraint)
    loss_c = -torch.log(F.sigmoid(kappa * (-f_p - f_q))).mean()

    # Total loss: Combine all components
    total_loss = loss_a + loss_b + loss_c 

    return total_loss