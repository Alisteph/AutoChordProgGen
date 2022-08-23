'''
ini_chordにコード名のリスト（最初のコード進行）を与えます．
chord_strsで生成されたコード進行がコード名のリストとして与えられます．
ini_chordには1つ以上のコード名を入れます．
'''

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
from funcs import *


def generate_chord(init_chords: list) -> list:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_lang = {"input_dim":12, "hidden_dim":128, "target_dim":12, "num_layers":5}   # クロマベクトルdim12
    model = GRU_LM(**d_lang)
    model.to(device)
    l_weight= par_load('data/LM1.9.pth')
    new_params = l_weight
    model.load_state_dict(new_params)

    #ini_chords = ['F', 'G', 'Esus4', 'Am']    # USER GIVEN

    cvs = LM(init_chords, model, device)

    onchords = []
    chord_strs = []
    for i in range(cvs.shape[-1]):
        chord = cv2c(cvs[:,i])    # returns list of possible chordname like [<Chord: Em7/D>, <Chord: G6/D>]
        if len(chord) == 0:
            break
        else:
            chord = chord[0]
            onchords.append(chord)    # for midi
        chord_str = chord.root + str(chord.quality)    # trans into str like 'Em7'
        chord_strs.append(chord_str)    # for display

    
    norm_chords = [Chord(c) for c in chord_strs]

    # create midi files in given directory
    create_midi(onchords, 'tmp/on_chord.mid')    # 転回系
    create_midi(norm_chords, 'tmp/chord.mid')    # 名前通り

    return chord_strs
