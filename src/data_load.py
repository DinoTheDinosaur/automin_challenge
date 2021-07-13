import glob
import os
import re
import torch

from fnmatch import fnmatch
from num2words import num2words



def clean_automin_summaries(text):
    text = text.lower()
    lines = re.findall('\n[0-9)● •\-* ]+([^\n]+)', text)
    if lines == []:
        lines = re.split('[.!?]', text)
    return '\n'.join(lines)


def clean_automin_transcripts(text):
    text = text.lower()
    text = text.replace(u'\xa0', u' ')
    text = re.sub('[(<]\S+[)>]', '', text)
    match = re.search('(?<=\s)[0-9]+(th|rd|nd)', text)
    while match !=  None:
        offset = match.start(0)
        onset = match.end(0)
        text = text[:offset] + \
        num2words(int(text[offset:onset-2]), to='ordinal') + \
        text[onset:]
        match = re.search('(?<=\s)[0-9]+(th|rd|nd)', text)
    match = re.search('(?<=\s)[0-9]+(?=[a-z\s])', text)
    while match !=  None:
        offset = match.start(0)
        onset = match.end(0)
        text = text[:offset] + \
        num2words(int(text[offset:onset])) + ' ' + \
        text[onset:]
        match = re.search('(?<=\s)[0-9]+(?=[a-z\s])', text)
    text = re.sub('[.!?]', '\n', text)
    text = re.sub('[^qwertyuiopasdfghjklzxcvbnm\-\'\s\[\]0-9]', '', text)
    text = re.sub('(?<!\w)-(?!\w)', '', text)
    text = re.sub('(?<!\w)-(?=\w)', '', text)
    text = re.sub('(?<=\w)-(?!\w)', '', text)
    text = re.sub('\s(uh|uhm|ehm)\s', ' ', text)
    text = re.sub(' \'m', '\'m', text)
    text = re.sub('[ ]*\n+[ ]*', '\n', text)
    text = re.sub('[ ]+', ' ', text)
    text = re.sub('^[ \-]+', '', text)
    text = re.sub('[ \-]+$', '', text)
    lines = [line for line in text.split('\n') if line != '']
    text = '\n'.join(lines)
    print('###')
    print(text)
    return text


def load_dataset_automin_test(config):
    path = config['test']['automin']['path']
    data_filename_pattern = config['test']['automin']['data']['filename_pattern']
    filenames = []
    texts = []
    for subfolder in os.listdir(path):
        cur_transcripts = []
        subdir = os.path.join(path, subfolder)
        for filename in os.listdir(subdir):
            if fnmatch(filename, data_filename_pattern):
                with open(os.path.join(subdir, filename)) as f:
                    filenames += [filename]
                    cur_transcripts += [
                        clean_automin_transcripts(f.read())
                    ]
        texts += cur_transcripts
    return (filenames, texts, None)


def load_dataset_automin_validation(config):
    path = config['validation']['automin']['path']
    data_filename_pattern = config['validation']['automin']['data']['filename_pattern']
    summaries_filename_pattern = config['validation']['automin']['summaries']['filename_pattern']
    filenames = []
    texts = []
    summaries = []
    for subfolder in os.listdir(path):
        cur_transcripts = []
        cur_summaries = []
        subdir = os.path.join(path, subfolder)
        for filename in os.listdir(subdir):
            if fnmatch(filename, data_filename_pattern):
                with open(os.path.join(subdir, filename)) as f:
                    filenames += [filename]
                    cur_transcripts += [
                        clean_automin_transcripts(f.read())
                    ]
            elif fnmatch(filename, summaries_filename_pattern):
                with open(os.path.join(subdir, filename)) as f:
                    cur_summaries += [
                        clean_automin_summaries(f.read())
                    ]
        texts += cur_transcripts
        summaries += [cur_summaries]
    return (filenames, texts, summaries)


def load_dataset_ICSI_test(config):
    # load paths to audio
    torch.set_num_threads(1)
    vad, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    (get_speech_ts,
     get_speech_ts_adaptive,
     _, read_audio,
     _, _, _) = utils

    filenames = []
    signals = []
    summaries = []

    for foldername in os.listdir(config['test']['ICSI']['data']['path']):
        cur_signals = []
        subfolder = os.path.join(
            config['test']['ICSI']['data']['path'],
            foldername
        )
        filenames += [subfolder]
        for filename in os.listdir(subfolder):
            full_filedir = os.path.join(
                subfolder,
                filename
            )
            if not fnmatch(filename, config['test']['ICSI']['data']['filename_pattern']):
                continue
            wav = read_audio(full_filedir)
            speech_timestamps = get_speech_ts(
                wav, vad
            )
            for timestamps in speech_timestamps:
                offset = timestamps['start']
                onset = timestamps['end']
                duration = (onset - offset) / 16000
                if duration > 60:
                    continue
                cur_signals += [(offset, wav[offset:onset])]
        signals += [list([signal for offset, signal in sorted(cur_signals, key=lambda x: x[0])])]

    return (filenames, signals, None)


DATA_LOADERS = {
    'validation': {
        'automin': load_dataset_automin_validation
    },
    'test': {
        'automin': load_dataset_automin_test,
        'ICSI': load_dataset_ICSI_test
    }
}