import sys

sys.path.insert(0, "../src/PreSumm/src")

from src.PreSumm.src.predict_presumm import predict

#you should change that part to whole dataset run
with open("../../../data/automin-2021-confidential-data-main/task-A-elitr-minuting-corpus-en/train/meeting_en_train_001/"
          "transcript_MAN2_annot02.deidentified.txt", "r") as f:
    text = f.read().strip()\
        .lower()\
        .replace('uhm', '') \
        .replace('uh', '') \
        .replace('ehm', '') \
        .replace('yeah', '')\
        .replace('okay', '') \
        .replace('yes', '')

print(len(text))
pred_text_list = predict([text])
print(pred_text_list)
print(len(pred_text_list[0]))