TOKEN_OFFSET = 100

import argparse
import contextlib
import json
import os
import pickle
import glob

import editdistance
import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import nemo
import nemo.collections.asr as nemo_asr
from src import kenlm_utils
from src.models import EncDecCTCModelBPEAutoMin
from nemo.utils import logging


def beam_search_eval(
    all_probs,
    vocab,
    ids_to_text_func=None,
    preds_output_file=None,
    lm_path=None,
    beam_alpha=1.0,
    beam_beta=0.0,
    beam_width=128,
    beam_batch_size=128,
    progress_bar=True,
    result_lines=[],
):
    # creating the beam search decoder
    beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
        vocab=vocab,
        beam_width=beam_width,
        alpha=beam_alpha,
        beta=beam_beta,
        lm_path=lm_path,
        num_cpus=max(os.cpu_count(), 1),
        input_tensor=False,
    )

    words_count = 0
    chars_count = 0
    sample_idx = 0
    if preds_output_file:
        out_file = open(preds_output_file, 'w')

    if progress_bar:
        it = tqdm(
            range(int(np.ceil(len(all_probs) / beam_batch_size))),
            desc=f"Beam search decoding with width={beam_width}, alpha={beam_alpha}, beta={beam_beta}",
            ncols=120,
        )
    else:
        it = range(int(np.ceil(len(all_probs) / beam_batch_size)))
    for batch_idx in it:
        # disabling type checking
        with nemo.core.typecheck.disable_checks():
            probs_batch = all_probs[batch_idx * beam_batch_size : (batch_idx + 1) * beam_batch_size]
            beams_batch = beam_search_lm.forward(log_probs=probs_batch, log_probs_length=None,)

        for beams_idx, beams in enumerate(beams_batch):
            beams = sorted(beams, key=lambda item: item[0], reverse=True)
            candidate = beams[0]
            pred_text = ids_to_text_func([ord(c) - TOKEN_OFFSET for c in candidate[1]])
            score = candidate[0]
            if preds_output_file:
                out_file.write('{}\t{}\t{}\n'.format(
                    result_lines[batch_idx*beam_batch_size + beams_idx],
                    pred_text,
                    score
                ))
        sample_idx += len(probs_batch)

    if preds_output_file:
        out_file.close()
        logging.info(f"Stored the predictions of beam search decoding at '{preds_output_file}'.")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate an ASR model with beam search decoding and n-gram KenLM language model.'
    )
    parser.add_argument(
        "--nemo_model_file", default='models/stt_en_conformer_ctc_large_ls.nemo', type=str, help="The path of the '.nemo' file of the ASR model"
    )
    parser.add_argument(
        "--kenlm_model_file", default='/home/dino/datasets/AMI/data/ami_public_manual_1.6.2/lm.bin', type=str, help="The path of the KenLM binary model file"
    )
    parser.add_argument("--audio_dir", default='/home/dino/datasets/ICSI/Signals_WAV/', type=str)
    parser.add_argument(
        "--preds_output_folder", default='results', type=str, help="The optional folder where the predictions are stored"
    )
    parser.add_argument(
        "--probs_cache_file", default='tmp/probs.cache', type=str, help="The cache file for storing the outputs of the model"
    )
    parser.add_argument(
        "--acoustic_batch_size", default=16, type=int, help="The batch size to calculate log probabilities"
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="The device to load the model onto to calculate log probabilities"
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Whether to use AMP if available to calculate log probabilities"
    )
    parser.add_argument(
        "--decoding_mode",
        choices=["greedy", "beamsearch", "beamsearch_ngram"],
        default="beamsearch_ngram",
        type=str,
        help="The decoding scheme to be used for evaluation.",
    )
    parser.add_argument(
        "--beam_width",
        default=128,
        type=int,
        nargs="+",
        help="The width or list of the widths for the beam search decoding",
    )
    parser.add_argument(
        "--beam_alpha",
        default=1.0,
        type=float,
        nargs="+",
        help="The alpha parameter or list of the alphas for the beam search decoding",
    )
    parser.add_argument(
        "--beam_beta",
        default=0.0,
        type=float,
        nargs="+",
        help="The beta parameter or list of the betas for the beam search decoding",
    )
    parser.add_argument(
        "--beam_batch_size", default=128, type=int, help="The batch size to be used for beam search decoding"
    )
    args = parser.parse_args()

    asr_model = EncDecCTCModelBPEAutoMin.restore_from(
        args.nemo_model_file, map_location=torch.device(args.device)
    )

    if args.use_amp:
        if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            logging.info("AMP is enabled!\n")
            autocast = torch.cuda.amp.autocast
    else:
        @contextlib.contextmanager
        def autocast():
            yield
    # load paths to audio
    filepaths = sorted(list(glob.glob(os.path.join(args.audio_dir, f"*/*.wav"))))
    torch.set_num_threads(1)
    vad, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    (get_speech_ts,
     get_speech_ts_adaptive,
     _, read_audio,
     _, _, _) = utils
    all_logits = []
    result_lines = []
    with autocast():
        with torch.no_grad():
            for i, filename in enumerate(filepaths):
                wav = read_audio(filename)
                speech_timestamps = get_speech_ts_adaptive(
                    wav, vad
                )
                for i, timestamps in enumerate(speech_timestamps):
                    offset = timestamps['start']
                    onset = timestamps['end']
                    filename_part = filename.replace('.wav', f'_{i}.wav')
                    all_logits += [asr_model.transcribe(
                        [filename_part],
                        [wav[offset:onset]],
                        logprobs=True
                    )[0]]
                    result_lines += [
                        f'{filename}\t{filename_part}\t{offset}\t{onset}'
                    ]
    all_probs = [kenlm_utils.softmax(logits) for logits in all_logits]
    vocab = asr_model.decoder.vocabulary
    ids_to_text_func = None
    vocab = [chr(idx + TOKEN_OFFSET) for idx in range(len(vocab))]
    ids_to_text_func = asr_model.tokenizer.ids_to_text
    del asr_model

    if args.decoding_mode == "beamsearch_ngram":
        if not os.path.exists(args.kenlm_model_file):
            raise FileNotFoundError(f"Could not find the KenLM model file '{args.kenlm_model_file}'.")
        lm_path = args.kenlm_model_file
    else:
        lm_path = None

    # 'greedy' decoding_mode would skip the beam search decoding
    if args.decoding_mode in ["beamsearch_ngram", "beamsearch"]:

        params = {'beam_width': [args.beam_width], 'beam_alpha': [args.beam_alpha], 'beam_beta': [args.beam_beta]}
        hp_grid = ParameterGrid(params)
        hp_grid = list(hp_grid)

        logging.info(f"==============================Starting the beam search decoding===============================")
        logging.info(f"Grid search size: {len(hp_grid)}")
        logging.info(f"It may take some time...")
        logging.info(f"==============================================================================================")

        if args.preds_output_folder and not os.path.exists(args.preds_output_folder):
            os.mkdir(args.preds_output_folder)
        for hp in hp_grid:
            if args.preds_output_folder:
                preds_output_file = os.path.join(
                    args.preds_output_folder,
                    f"preds_out_width{hp['beam_width']}_alpha{hp['beam_alpha']}_beta{hp['beam_beta']}.tsv",
                )
            else:
                preds_output_file = None

            beam_search_eval(
                all_probs=all_probs,
                vocab=vocab,
                ids_to_text_func=ids_to_text_func,
                preds_output_file=preds_output_file,
                lm_path=lm_path,
                beam_width=hp["beam_width"],
                beam_alpha=hp["beam_alpha"],
                beam_beta=hp["beam_beta"],
                beam_batch_size=args.beam_batch_size,
                progress_bar=True,
                result_lines=result_lines,
            )


if __name__ == '__main__':
    main()