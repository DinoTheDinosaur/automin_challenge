import glob
from pyrouge import Rouge155

from nltk.tokenize import sent_tokenize, word_tokenize

import nltk
# nltk.download('punkt')
from train_abstractive import test_text_abs
import os.path


DIR_DATA = '../data_contest/automin-2021-confidential-data-main/task-A-elitr-minuting-corpus-en/train/'


# def main():
#     for file in glob.glob(DIR_DATA + "**/*deidentified.txt"):
#         print(file)

num_sentence = 25
def main():
    file_origin_list = []
    for file in glob.glob(DIR_DATA + "**/transcript_MAN*.deidentified.txt"):
        sum_file = '/'.join(file.split('/')[:-1]) + '/' + 'minutes_ORIG.deidentified.txt'
        if os.path.isfile(sum_file):
            file_origin_list.append((file, sum_file))
            print(sum_file)

    # файл с удаленными переносами
    file_name = file_origin_list[0][0]
    sum_file_name = file_origin_list[0][1]
    print(file_name)

    with open(file_name, 'r') as file:
        data = file.read().replace('\n', '')


    # разбитие на num_sentence предлоений для кусочной обработки
    sent_tokenize_list = data.split('.')

    import argparse
    args = argparse.Namespace()
    args.test_from = '../models/cnndm_baseline_best.pt'
    args.visible_gpus = 0
    args.large = False
    args.temp_dir = '../temp'
    args.finetune_bert = False
    args.enc_dropout = 0.2
    args.max_pos = 512
    args.share_emb = False
    args.dec_heads = 8
    args.dec_dropout = 0.2
    args.text_src = '../test'
    args.text_tgt = ''
    args.alpha = 0.6
    args.beam_size = 5
    args.min_length = 15
    args.max_length = 150
    args.max_tgt_len = 140
    args.model_path = '../models/'
    args.result_path = '../results/cnndm'
    args.recall_eval = False
    args.block_trigram = True
    for n in range(0, len(sent_tokenize_list)-num_sentence, num_sentence):
        # print(n)
        # print(n+num_sentence)
        print(len(' '.join(sent_tokenize_list[n:n+num_sentence]).split(' ')))
        print(' '.join(sent_tokenize_list[n:n+num_sentence]))
        print('-------------------')
        # args = {'test_from': '../models/cnndm_baseline_best.pt'}

        # parser.add_argument("-result_path", default='../results/cnndm')
        # parser.add_argument("-temp_dir", default='../temp')
        # parser.add_argument("-text_src", default='')
        # parser.add_argument("-text_tgt", default='')
        test_text_abs(args)

        with open('../results/cnndm.-1.candidate', 'r') as file:
            data = file.read().replace('\n', '')
            print(data)


        # .../data_contest/automin-2021-confidential-data-main/task-A-elitr-minuting-corpus-en/train/meeting_en_train_023/transcript_MAN_annot07.deidentified.txt
        part1_file_name = file_name.split('/')[-1][:-4]
        part2_file_name = file_name.split('/')[-2]

        num_file_name = part2_file_name.split('_')[-1]
        part2_file_name = part2_file_name[:-4]

        new_file_name = part1_file_name + '__' + part2_file_name + '.' + str(n) + num_file_name + '.txt'
        new_model_file_name = new_file_name[:-4].split('.')[0] + '.' + new_file_name[:-4].split('.')[1] \
                              + '.A.' + str(n) + num_file_name  +  new_file_name[-4:]
        print(new_file_name)
        print(new_model_file_name)

        print(sum_file_name.split('/')[-1])

        from shutil import copyfile
        copyfile(sum_file_name, '../results/summaries/' + new_file_name)

        # with open('../results/model_summaries/' + sum_file_name.split('/')[-1], 'w+') as f:
        #     f.write(data)

        with open('../results/' + new_model_file_name, 'w') as f1:
            f1.write(data)

        break

        #save file result
    r = Rouge155()
    r.system_dir = '../results/summaries'
    r.model_dir = '../results/model_summaries'
    # передвинуть папку с именем в конец - заменить _ перед 001 на .
    # minutes_ORIG.deidentified__meeting_en_train.001.txt
    r.system_filename_pattern = 'transcript_MAN_annot07.deidentified__meeting_en_train.(\d+).txt'
    # добавить между именем A
    # minutes_ORIG.deidentified__meeting_en_train.A.001.txt
    r.model_filename_pattern = 'transcript_MAN_annot07.deidentified__meeting_en_train.A.#ID#.txt'

    output = r.convert_and_evaluate()
    print(output)
    output_dict = r.output_to_dict(output)

# if __name__ == "__main__":
main()
