import re
import sys
# sys.path.insert(0, "../src/PreSumm/src")
import argparse

import nltk
import torch
from pytorch_transformers import BertTokenizer

from src.PreSumm.src.train_abstractive import test_text_abs

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']
import os
os.chdir('../src/PreSumm/src')

def predict(texts):

    args = argparse.Namespace()
    args.test_from = '../models/cnndm_baseline_best.pt'
    args.visible_gpus = 0
    args.large = False
    args.temp_dir = 'temp/'
    args.finetune_bert = False
    args.enc_dropout = 0.2
    args.max_pos = 512
    args.share_emb = False
    args.dec_heads = 8
    args.dec_dropout = 0.2
    args.text_src = 'test'
    args.text_tgt = ''
    args.alpha = 0.4
    args.beam_size = 5
    args.min_length = 20
    args.max_length = 100
    args.max_tgt_len = 140
    args.model_path = '../models'
    args.result_path = '../models/cnndm'
    args.recall_eval = False
    args.block_trigram = True

    NUM_WORDS = 512
    minutes = []
    import os
    # cwd = os.getcwd()

    print(sys.path)

    for text in texts:
        parts = []
        sentences = nltk.word_tokenize(text)

        for i in range(len(sentences)//NUM_WORDS):
            if i != (len(sentences)//NUM_WORDS -1):
                part = ' '.join(sentences[i*NUM_WORDS : (i+1) * NUM_WORDS])
            else:
                part = ' '.join(sentences[i * NUM_WORDS: ])
            parts.append(part)
        num_original = len(sentences)
        num = num_original

        while num >= int(0.1 * num_original):
            print("NEW ITERATION")
            res = ''
            for part in parts:
                with open("test", "w") as f:
                    f.write("%s \n" % (part))
                minute = test_text_abs(args)
                print("ORIGINS: %s" % (part))
                print("HYPOTHESIS: %s" % (minute))
                res += ' ' + minute
            sentences = nltk.word_tokenize(res)
            parts = []
            print(len(sentences))
            for i in range(len(sentences) // NUM_WORDS):
                if i != (len(sentences) // NUM_WORDS - 1):
                    part = ' '.join(sentences[i * NUM_WORDS: (i + 1) * NUM_WORDS])
                else:
                    part = ' '.join(sentences[i * NUM_WORDS:])
                part = re.sub(r'\b(\w+)( \1\b)+', r'\1', part)
                parts.append(part)

            if not parts:
                break

            num = len(sentences)

        minute = ' '.join(sentences)
        minutes.append(minute)
        print("RESULT: %s" % (minute))

    return minutes

# TEST input text - return minute
