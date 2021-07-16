# automin_challenge

Pipeline for automatic minuting for text transcriptions and audio recordings

## Installation

The following pipeline was tested in Python 3.8 only. To run the package clone the repository:
```
git clone https://github.com/DinoTheDinosaur/automin_challenge.git
```
Install the dependencies for the pipeline
```
cd automin_challenge
pip install -r requirements.txt
```
If you wish to run the pipeline on audio data, download an acoustic and language models and place them in subfolder `models/`. Acoustic model download is available at `https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_large`, while pretrained language model is available at `https://drive.google.com/file/d/1ENH1icE19yKcKz0e8wIJ0fpXVJmRQhol/view?usp=sharing`. After placing all the models in the folder you should be able to see the following:
```
$ cd models
$ ls
lm.bin  README.md  stt_en_conformer_ctc_large_ls.nemo
```

## Run the pipeline

Main file for defining the running mode for the pipeline is `config/default_config.yaml`. Most of the parameters are correctly pre-selected, you might want to modify the mode and dataset name (first three lines of default_config.yaml):
```
mode: validation
dataset_name: automin
summarization: cluster
```
If mode `validation` is chosen, metrics will be computed for the predicts and summaries will be written to ouput files. If mode `test` is chosen, only summaries will be generated. Apart from the `automin` dataset, the pipeline is also adapted for running on ICSI dataset.

Besides all above, it is possible to choose an algorithm for summary generation. The available variants:
- `random` (baseline): select 10 random lines as a summary
- `t5`: usage of t5 model for summarization
- `pegasus`: usage of pegasus model for summarization
- `cluster`: usage of USE vectorisation & clustering approaches for summary constructing

To run the pipeline launch from folder automin_challenge:
```
python -m src.run_inference
```
Different datasets have different in-built dataset readers and results writers, so if you wish to generate minutes for a different type of dataset, you will need to define custom readers and writers for the datasets.

Do not forget to modify dataset paths once you decide to run the pipeline on a particular dataset:
```
path: /path/to/your/automin-2021-confidential-data/task-A-elitr-minuting-corpus-en/dev
```

## Results

We've precomputed results for each of the approaches for summarization:
| Metric | random | t5 | pegasus | cluster |
| -------|--------|----|---------|---------|
| rouge-1 Recall | 0.13 | 0.1 | 0.19 | 0.24 |
| rouge-1 Precision | 0.24 | 0.23 | 0.21 | 0.18 |
| rouge-1 F1 | 0.17 | 0.14 | 0.20 | 0.21 |
| rouge-2 Recall | 0.01 | 0.01 | 0.02 | 0.02 |
| rouge-2 Precision | 0.02 | 0.02 | 0.03 | 0.02 |
| rouge-2 F1 | 0.01 | 0.01 | 0.03 | 0.02 |
| rouge-4 Recall | 0.0 | 0.0 | 0.00 | 0.00 |
| rouge-4 Precision | 0.0 | 0.0 | 0.01 | 0.00 |
| rouge-4 F1 | 0.0 | 0.0 | 0.00 | 0.00 |
| rouge-l Recall | 0.12 | 0.09 | 0.16 | 0.23 |
| rouge-l Precision | 0.23 | 0.21 | 0.19 | 0.18 |
| rouge-l F1 | 0.16 | 0.13 | 0.17 | 0.20 |
| rouge-w-1.2 Recall | 0.04 | 0.03 | 0.06 | 0.08 |
| rouge-w-1.2 Precision | 0.14 | 0.13 | 0.11 | 0.10 |
| rouge-w-1.2 F1 | 0.07 | 0.06 | 0.08 | 0.09 |
| rouge-s4 Recall | 0.01 | 0.01 | 0.02 | 0.02 |
| rouge-s4 Precision | 0.03 | 0.03 | 0.03 | 0.02 |
| rouge-s4 F1 | 0.02 | 0.01 | 0.03 | 0.02 |
| rouge-su4 Recall | 0.03 | 0.02 | 0.05 | 0.06 |
| rouge-su4 Precision | 0.06 | 0.06 | 0.06 | 0.05 |
| rouge-su4 F1 | 0.04 | 0.03 | 0.06 | 0.05 |
| BERT Recall | 0.80 | 0.8 | 0.79 | 0.79 |
| BERT Precision | 0.77 | 0.75 | 0.76 | 0.78 |
| BERT F1 | 0.77 | 0.76 | 0.76 | 0.77 |
| Sentence Mover Score* | 1. | 1. | 1. | 1. |

*SMS metric seems to be bugged, took from original repository https://github.com/eaclark07/sms with no modifications

## Authors

- [Olga Iakovenko](https://github.com/DinoTheDinosaur)
- [Anna Andreeva](https://github.com/rogotulka)
- [Anna Lapidus](https://github.com/AnyaLa)
- [Liana Mikaelyan](https://github.com/LianaMikael)

## PS

We're looking towards making our system better, so all contributions and comments are highly appreciated! Feel free to suggest any improvements you feel would be useful for this system. Thank you!
