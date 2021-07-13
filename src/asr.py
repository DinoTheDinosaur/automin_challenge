TOKEN_OFFSET = 100

import contextlib
import nemo
import os
import torch

import numpy as np

from src.models import EncDecCTCModelBPEAutoMin


def beam_search_eval(
    all_probs,
    vocab,
    ids_to_text_func=None,
    lm_path=None,
    beam_alpha=2.0,
    beam_beta=0.0,
    beam_width=128,
    beam_batch_size=128,
):
    # creating the beam search decoder
    beam_search_lm = nemo.collections.asr.modules.BeamSearchDecoderWithLM(
        vocab=vocab,
        beam_width=beam_width,
        alpha=beam_alpha,
        beta=beam_beta,
        lm_path=lm_path,
        num_cpus=max(os.cpu_count(), 1),
        input_tensor=False,
    )

    results = []

    for audio_file in all_probs:
        lines = []
        it = range(int(np.ceil(len(audio_file) / beam_batch_size)))
        for batch_idx in it:
            # disabling type checking
            with nemo.core.typecheck.disable_checks():
                probs_batch = audio_file[batch_idx * beam_batch_size : (batch_idx + 1) * beam_batch_size]
                beams_batch = beam_search_lm.forward(log_probs=probs_batch, log_probs_length=None,)

            for beams_idx, beams in enumerate(beams_batch):
                beams = sorted(beams, key=lambda item: item[0], reverse=True)
                candidate = beams[0]
                pred_text = ids_to_text_func([ord(c) - TOKEN_OFFSET for c in candidate[1]])
                lines += [pred_text]
        results += ['\n'.join(lines)]
    return results


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])


def predict_text(audio_data, config):
    @contextlib.contextmanager
    def autocast():
        yield

    asr_model = EncDecCTCModelBPEAutoMin.restore_from(
        config['am_path'], map_location=torch.device(config['device'])
    )
    all_probs = []

    with autocast():
        with torch.no_grad():
            for meeting_recording in audio_data:
                all_probs += [[]]
                for i, meeting_phrase in enumerate(meeting_recording):
                    all_probs[-1] += [softmax(asr_model.transcribe(
                        [f'{i}.wav'],
                        [meeting_phrase],
                        logprobs=True
                    )[0])]
    vocab = asr_model.decoder.vocabulary
    ids_to_text_func = None
    vocab = [chr(idx + TOKEN_OFFSET) for idx in range(len(vocab))]
    ids_to_text_func = asr_model.tokenizer.ids_to_text
    del asr_model

    lm_path = config['lm_path']
    result_texts = beam_search_eval(
        all_probs=all_probs,
        vocab=vocab,
        ids_to_text_func=ids_to_text_func,
        lm_path=lm_path,
        beam_batch_size=config['beam_batch_size']
    )
    return result_texts