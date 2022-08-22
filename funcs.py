import os
import numpy as np
import sys
import glob
import random
import torch
from torch import nn
import torch.nn.functional as F
import math
import time
import datetime
import json
import pretty_midi
from pychord import Chord, find_chords_from_notes


class GRU_LM(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_dim, num_layers):
        super (GRU_LM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.ln1 = nn.Linear(self.input_dim, self.input_dim*4)
        self.gru = nn.GRU(self.input_dim*8, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.1)
        self.hidden2target  = nn.Linear(self.hidden_dim, self.target_dim)
        self.sigmoid = nn.Sigmoid()

        self.conv11 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=(1,1))
        self.conv12 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(16, affine=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4,1))
        self.conv21 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1,1))
        self.conv22 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(32, affine=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4,1))

    def extract_feature(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.bn2(x)
        x = self.pool2(x)
        return x

    def forward(self, x):
        '''
            batch * chroma vec * length
            B * 12 * T -> B * 12 * T
        '''
        x = x.permute(0,2,1)
        x = self.ln1(x)
        x = x.permute(0,2,1)
        x = torch.unsqueeze(x, 1)
        x = self.extract_feature(x)
        x = self.dropout1(x)
        x = x.view(-1, x.size(1)*x.size(2), x.size(3)).permute(0,2,1)
        x = F.elu(x)
        self.gru.flatten_parameters()
        x, h_n = self.gru(x)
        x = self.dropout2(x)
        x = self.hidden2target(x).permute(0,2,1)
        x = self.sigmoid(x)
        return x


def LM_train(x, model, device):
    x_mov = torch.cat([torch.zeros(x.size(0),x.size(1),1).to(device), x[:,:,:-1]], axis=2).float().to(device)
    y = model(x_mov)
    x = x.permute(0,2,1)
    y = y.permute(0,2,1)
    criterion = nn.functional.binary_cross_entropy
    return criterion(y,x)


def key2Cmaj(key, chord):
    '''
        Chord, list(Chord)
        do not distinguish major and minor
        -> wanna do...
    '''
    for tran in range(12):
        if Chord(key.root) == Chord('C'): break
        key.transpose(1)
    if str(key.quality) == 'm': tran -= 3
    tran = tran % 12
    for c in chord: c.transpose(tran)
    chord_modified = chord.copy()
    return chord_modified


def note2cv(note_chord):
    for i in range(12):
        note_number = Chord('C')
        note_number.transpose(i)
        if Chord(note_chord) == note_number: break
    cv = np.zeros((12,1))
    cv[i, 0] = 1
    return cv


def c2cv(chord):
    '''
        chord -> chroma vector <ndarray>
    '''
    note_list = chord.components()
    cv = np.zeros((12,1))
    for note in note_list:
        cv += note2cv(note)
    return cv


def cv2c(chord):
    '''
        chroma vector <ndarray> -> chord <Chord>
    '''
    note_list = []
    notes = np.where(chord==1)[0]    # np.where returns taple
    if notes.shape[0] == 0:
        note_list = [Chord('C').root]    # if notes has no notes it return C only
    else:
        for i in range(notes.shape[0]):
            n = Chord('C')
            n.transpose(int(notes[i]))
            note_list.append(n.root)
    return find_chords_from_notes(note_list)


def random_sample_file(INPUT_DIR, batchsize):
    files = glob.glob(INPUT_DIR + '/*_b.json')
    random_sample_file = random.sample(files, batchsize)
    songs = []
    for i, file in enumerate(random_sample_file):
        try:
            with open(file) as f:
                songs.append(json.load(f))
        except:
            print('Error in random_sample_file')
    return songs


def load_chords(batchsize):
    '''
        return Chord, list(Chord)
    '''
    path = '/content/drive/MyDrive/Colab Notebooks/0sh/chord_gen/song_json/song_json_b'
    songs = random_sample_file(path, batchsize)
    keys = []
    chords = []
    for i in range(len(songs)):
        try:    # .split()[0]の理由は'Play':'Em　　＜標準コード譜ページに戻る＞'とかあるから
            if songs[i]['Play'].split()[0] == 'None' or songs[i]['Play'].split()[0] == 'm': continue
            if len(songs[i]['Chords'].split()) < 10: continue # over 10 chords sequence 
            keys.append(Chord(songs[i]['Play'].split()[0]))
            chords.append([Chord(c) for c in songs[i]['Chords'].split()])
        except:
            print('This has Play: ' + songs[i]['Play'])
    return keys, chords


def par_load(dir):
    nn_par_path = dir
    load_weights = torch.load(nn_par_path)
    return load_weights


def gumbel_sigmoid(phi):
    u_1 = torch.rand(phi.size(), device="cuda")
    u_2 = torch.rand(phi.size(), device="cuda")
    noise = torch.log(torch.log(u_2 + 1e-20) / torch.log(u_1 + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((phi + noise) / 0.2)
    y_hard = (y_soft > 0.2).to(torch.float32)
    y = (y_hard - y_soft).detach() + y_soft
    # return y
    return y_soft


def LM(ini_chords_str, L_model, device):

    ini_chords = [Chord(c) for c in ini_chords_str]
    ini_chromas = [c2cv(c) for c in ini_chords]
    s = np.concatenate(ini_chromas, 1)

    s = np.array(s).astype(np.float32)
    s = s[np.newaxis,:,:]
    s = torch.tensor(s).to(device)
    s_mov = s

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    for i in range(32):
        y = L_model(s_mov)
        s_mov = torch.cat([s, torch.where(y<0.4, 0, 1)[:,:,s.size(-1)-1:]], axis=2).float().to(device)
    print(torch.where(s_mov<0.4, 0, 1).int().to('cpu').detach().numpy()[0])

    return torch.where(s_mov<0.4, 0, 1).int().to('cpu').detach().numpy()[0]