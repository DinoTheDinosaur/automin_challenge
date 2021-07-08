import random

def predict_summary(texts):
    ### PLACEHOLDER ###
    result = []
    for text in texts:
        lines = text.split('\n')
        random_lines = random.sample(lines, 10)
        result += ['\n'.join(random_lines)]
    return result