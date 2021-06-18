import os
import json
import torch
import tempfile

from typing import Callable, Dict, List, Optional, Union

from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.core.neural_types import *
from nemo.utils import logging

from omegaconf import DictConfig

class AudioSegmentAutoMin(AudioSegment):
    @classmethod
    def from_samples(
        cls, filename, samples, target_sr=16000, int_values=False, offset=0, duration=0, trim=False, orig_sr=16000,
    ):
        dtype = samples.dtype
        samples = samples.numpy()
        return cls(samples, 16000, target_sr=target_sr, trim=trim, orig_sr=orig_sr)

    @property
    def sample_rate(self):
        return self._sample_rate


class AudioToBPEDatasetAutoMin(_AudioTextDataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        samples,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        use_start_end_token: bool = True,
    ):
        if use_start_end_token and hasattr(tokenizer, 'bos_token'):
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, 'eos_token'):
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, 'pad_token'):
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                self._tokenizer = tokenizer

            def __call__(self, text):
                t = self._tokenizer.text_to_ids(text)
                return t
        self.samples = samples
        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            trim=trim,
        )
        self.featurizer = WaveformFeaturizerAutoMin(
            samples, sample_rate=sample_rate, int_values=int_values, augmentor=augmentor
        )

    def __getitem__(self, index):
        sample = self.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0
        features = self.featurizer.process(
            sample.audio_file, self.samples[index], offset=offset, duration=sample.duration, trim=self.trim, orig_sr=sample.orig_sr
        )
        f, fl = features, torch.tensor(features.shape[0]).long()

        t, tl = sample.text_tokens, len(sample.text_tokens)
        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

        return output



class WaveformFeaturizerAutoMin(WaveformFeaturizer):
    def __init__(self, samples, sample_rate=16000, int_values=False, augmentor=None):
        self.samples = samples
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.sample_rate = sample_rate
        self.int_values = int_values

    def process(self, filename, samples, offset=0, duration=0, trim=False, orig_sr=None):
        audio = AudioSegmentAutoMin.from_samples(
            filename,
            samples,
            target_sr=self.sample_rate,
            int_values=self.int_values,
            offset=offset,
            duration=duration,
            trim=trim,
            orig_sr=orig_sr,
        )
        return self.process_segment(audio)



class EncDecCTCModelBPEAutoMin(EncDecCTCModelBPE):

    def _setup_dataloader_from_config(self, samples, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None
        shuffle = config['shuffle']
        dataset = get_bpe_dataset(
            samples, config=config, tokenizer=self.tokenizer, augmentor=augmentor
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            num_workers=config.get('num_workers', 0),
            shuffle=config.get('shuffle', False),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_transcribe_dataloader(self, samples, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.
        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': min(config['batch_size'], len(config['paths2audio_files'])),
            'shuffle': False,
        }

        temporary_datalayer = self._setup_dataloader_from_config(
            samples, DictConfig(dl_config)
        )
        return temporary_datalayer

    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        samples,
        batch_size: int = 1,
        logprobs: bool = False,
        return_hypotheses: bool = False,
    ) -> List[str]:
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoder modules
            self.encoder.freeze()
            self.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                        fp.write(json.dumps(entry) + '\n')
                config = {'paths2audio_files': paths2audio_files, 'batch_size': batch_size, 'temp_dir': tmpdir}
                temporary_datalayer = self._setup_transcribe_dataloader(samples, config)
                for test_batch in temporary_datalayer:
                    logits, logits_len, greedy_predictions = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
                    if logprobs:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            lg = logits[idx][: logits_len[idx]]
                            hypotheses.append(lg.cpu().numpy())
                    else:
                        current_hypotheses = self._wer.ctc_decoder_predictions_tensor(
                            greedy_predictions, predictions_len=logits_len, return_hypotheses=return_hypotheses,
                        )

                        if return_hypotheses:
                            # dump log probs per file
                            for idx in range(logits.shape[0]):
                                current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                        hypotheses += current_hypotheses

                    del greedy_predictions
                    del logits
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.decoder.unfreeze()
            logging.set_verbosity(logging_level)
        return hypotheses



def get_bpe_dataset(
    samples, config: dict, tokenizer: 'TokenizerSpec', augmentor: Optional['AudioAugmentor'] = None
) -> AudioToBPEDatasetAutoMin:
    """
    Instantiates a Byte Pair Encoding / Word Piece Encoding based AudioToBPEDataset.
    Args:
        config: Config of the AudioToBPEDataset.
        tokenizer: An instance of a TokenizerSpec object.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.
    Returns:
        An instance of AudioToBPEDataset.
    """
    dataset = AudioToBPEDatasetAutoMin(
        samples,
        manifest_filepath=config['manifest_filepath'],
        tokenizer=tokenizer,
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        trim=config.get('trim_silence', False),
        use_start_end_token=config.get('use_start_end_token', True),
    )
    return dataset
