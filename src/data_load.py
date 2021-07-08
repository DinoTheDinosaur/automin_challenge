import os
import re

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
    text = re.sub('[^qwertyuiopasdfghjklzxcvbnm\-\'\s]', '', text)
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
    return '\n'.join(lines)



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



DATA_LOADERS = {
    'validation': {
        'automin': load_dataset_automin_validation
    },
    'test': {
        'automin': load_dataset_automin_test
    }
}