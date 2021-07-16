import argparse
import os
import yaml

from pathlib import Path
from src.asr import predict_text
from src.data_load import DATA_LOADERS
from src.evaluation import evaluate
from src.summarization import predict_clustering



def main():
    parser = argparse.ArgumentParser(
        description='Inference for the AutoMin 2021 challenge'
    )
    parser.add_argument(
        "--config", default='config/default_config.yaml',
        type=str, help="Path to the config file"
    )
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    (dataset_path, filenames,
    data, summary_variants) = DATA_LOADERS[config['mode']][config['dataset_name']](config)

    if type(data[0]) != str:
        texts = predict_text(data, config)
    else:
        texts = data

    predicts = predict_clustering(texts)

    if summary_variants:
        scores = evaluate(summary_variants, predicts)
        print(scores)

    if config['dataset_name'] == 'automin':
        result_folder = str(Path(dataset_path) / '..' / '..' / 'submission-folder' / 'task-A' / 'en')
        for filename, predict in zip(filenames, predicts):
            result_filename = str(Path(filename.replace(
                dataset_path, result_folder
            )).parent) + '.txt'
            os.makedirs(Path(result_filename).parent, exist_ok=True)
            with open(result_filename, 'w') as f:
                f.write(predict)
    else:
        with open(config['results_path'], 'w') as f:
            blocks = [f'{filename}\n{predict}' for filename, predict in zip(filenames, predicts)]
            results_text = '\n\n###\n\n'.join(list(blocks))
            f.write(results_text)


if __name__ == '__main__':
    main()