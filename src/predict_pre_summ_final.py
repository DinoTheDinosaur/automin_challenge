import os
# import os
import re

print(os.path.abspath(os.getcwd()))

import sys

sys.path.insert(0, "../src/PreSumm/src")


from src.PreSumm.src.predict_presumm import predict
import glob

for f in glob.glob('/'.join(os.path.abspath(os.getcwd()).split('/')[:-3]) +
                       '/data/automin-2021-confidential-data-main/task-A-elitr-minuting-corpus-en/test_II/*/*.txt'):

    if f.endswith(".txt"):
        file_path = os.path.join(f)
        save_file_name = '/'.join(os.path.abspath(os.getcwd()).split('/')[:-3]) +\
                         "/data/automin-2021-confidential-data-main/sample-submission-folder/task-A/en/" \
                         + file_path.split('/')[-2] + '.txt'

        save_file_name = '/'.join(os.path.abspath(os.getcwd()).split('/')[:-3]) + \
                         "/data/automin-2021-confidential-data-main/sample-submission-folder/task-A/en/" \
                         + file_path.split('/')[-2] + '.txt'
        print(save_file_name)
        with open(file_path, "r") as f:
            text = f.read().strip() \
                .lower() \
                .replace('uhm', '') \
                .replace('uh', '') \
                .replace('ehm', '') \
                .replace('yeah', '') \
                .replace('okay', '') \
                .replace('okey', '') \
                .replace('yes', '')\
                .replace('bye', '')\
                .replace('sorry', '')
        pred_text_list = predict([text])

        with open(save_file_name, "w") as f:
            pred_text = pred_text_list[0]
            pred_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', pred_text)
            f.write(pred_text)

