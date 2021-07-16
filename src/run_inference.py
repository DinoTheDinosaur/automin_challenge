import argparse
import yaml

from src.asr import predict_text
from src.data_load import DATA_LOADERS
from src.evaluation import evaluate
from src.summarization import SUMMARIZATIONS
from src.write_results import RESULT_WRITERS



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

    (filenames, data, summary_variants) = DATA_LOADERS[config['mode']][config['dataset_name']](config)

    if type(data[0]) != str:
        texts = predict_text(data, config)
    else:
        texts = data

    predicts = SUMMARIZATIONS[config['summarization']](texts)

    if summary_variants:
        scores = evaluate(summary_variants, predicts)
        print(scores)

    RESULT_WRITERS[config['dataset_name']](
        filenames, predicts, config
    )



if __name__ == '__main__':
    main()