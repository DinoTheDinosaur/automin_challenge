import argparse
import yaml

from src.asr import predict_text
from src.data_load import DATA_LOADERS
from src.evaluation import evaluate
from src.summarization import predict_summary



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

    filenames, data, summary_variants = DATA_LOADERS[config['mode']][config['dataset_name']](config)

    if type(data[0]) != str:
        texts = predict_text(data, config)
    else:
        texts = data

    predicts = predict_summary(texts)

    if summary_variants:
        scores = evaluate(summary_variants, predicts)
        print(scores)

    with open(config['results_path'], 'w') as f:
        blocks = [f'{filename}\n{predict}' for filename, predict in zip(filenames, predicts)]
        results_text = '\n\n###\n\n'.join(list(blocks))
        f.write(results_text)


if __name__ == '__main__':
    main()