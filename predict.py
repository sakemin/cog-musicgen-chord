# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import random

# We need to set `TRANSFORMERS_CACHE` before any imports, which is why this is up here.
MODEL_PATH = "/src/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH

from typing import Optional
from cog import BasePredictor, Input, Path

# Model specific imports
import torchaudio
import typing as tp
import numpy as np

import torch

from audiocraft.solvers.compression import CompressionSolver

from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.solvers.compression import CompressionSolver
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
)
from audiocraft.data.audio import audio_write

from audiocraft.models.builders import get_lm_model
from omegaconf import OmegaConf

import subprocess

def _delete_param(cfg, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)

def load_ckpt(path, device, url=False):
    if url:
        loaded = torch.hub.load_state_dict_from_url(str(path))
    else:
        loaded = torch.load(str(path))
    cfg = OmegaConf.create(loaded['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_chord.cache_path')
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    lm = get_lm_model(loaded['xp.cfg'])
    lm.load_state_dict(loaded['model']) 
    lm.eval()
    lm.cfg = cfg
    compression_model = CompressionSolver.model_from_checkpoint(cfg.compression_model_checkpoint, device=device)
    return MusicGen(f"{os.getenv('COG_USERNAME')}/musicgen-chord", compression_model, lm)

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.mbd = MultiBandDiffusion.get_mbd_musicgen()
        
        if str(weights) == "weights":
            weights = None

        if weights is not None:
            print("Fine-tuned model weights loaded!")
            self.model = load_ckpt(weights, self.device, url=True)

    def _load_model(
        self,
        model_path: str,
        cls: Optional[any] = None,
        load_args: Optional[dict] = {},
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> MusicGen:

        if device is None:
            device = self.device

        compression_model = load_compression_model(
            model_id, device=device, cache_dir=model_path
        )
        lm = load_lm_model(model_id, device=device, cache_dir=model_path)
        
        return MusicGen(model_id, compression_model, lm)

    def predict(
        self,
        model_version: str = Input(
            description="Model type. Select `fine-tuned` if you trained the model into your own repository.", default="stereo-chord-large",
            choices=["chord", "chord-large", "stereo-chord", "stereo-chord-large", "fine-tuned"]
        ),
        prompt: str = Input(
            description="A description of the music you want to generate.", default=None
        ),
        text_chords: str = Input(
            description="A text based chord progression condition. Single uppercase alphabet character(eg. `C`) is considered as a major chord. Chord attributes like(`maj`, `min`, `dim`, `aug`, `min6`, `maj6`, `min7`, `minmaj7`, `maj7`, `7`, `dim7`, `hdim7`, `sus2` and `sus4`) can be added to the root alphabet character after `:`.(eg. `A:min7`) Each chord token splitted by `SPACE` is allocated to a single bar. If more than one chord must be allocated to a single bar, cluster the chords adding with `,` without any `SPACE`.(eg. `C,C:7 G, E:min A:min`) You must choose either only one of `audio_chords` below or `text_chords`.", default=None
        ),
        bpm: float = Input(
            description="BPM condition for the generated output. `text_chords` will be processed based on this value. This will be appended at the end of `prompt`.", default=None
        ),
        time_sig: str = Input(
            description="Time signature value for the generate output. `text_chords` will be processed based on this value. This will be appended at the end of `prompt`.", default="4/4"
        ),
        audio_chords: Path = Input(
            description="An audio file that will condition the chord progression. You must choose only one among `audio_chords` or `text_chords` above.",
            default=None,
        ),
        audio_start: int = Input(
            description="Start time of the audio file to use for chord conditioning.",
            default=0,
            ge=0,
        ),
        audio_end: int = Input(
            description="End time of the audio file to use for chord conditioning. If None, will default to the end of the audio clip.",
            default=None,
            ge=0,
        ),
        duration: int = Input(
            description="Duration of the generated audio in seconds.", default=8
        ),
        continuation: bool = Input(
            description="If `True`, generated music will continue from `audio_chords`. If chord conditioning, this is only possible when the chord condition is given with `text_chords`. If `False`, generated music will mimic `audio_chords`'s chord.",
            default=False,
        ),
        # continuation_start: int = Input(
        #     description="Start time of the audio file to use for continuation.",
        #     default=0,
        #     ge=0,
        # ),
        # continuation_end: int = Input(
        #     description="End time of the audio file to use for continuation. If -1 or None, will default to the end of the audio clip.",
        #     default=None,
        #     ge=0,
        # ),
        multi_band_diffusion: bool = Input(
            description="If `True`, the EnCodec tokens will be decoded with MultiBand Diffusion. Not compatible with stereo models.",
            default=False,
        ),
        normalization_strategy: str = Input(
            description="Strategy for normalizing audio.",
            default="loudness",
            choices=["loudness", "clip", "peak", "rms"],
        ),
        chroma_coefficient: float = Input(
            description="Coefficient value multiplied to multi-hot chord chroma.",
            default=1.0,
            ge=0.5,
            le=2.5
        ),
        top_k: int = Input(
            description="Reduces sampling to the k most likely tokens.", default=250
        ),
        top_p: float = Input(
            description="Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used.",
            default=0.0,
        ),
        temperature: float = Input(
            description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.",
            default=1.0,
        ),
        classifier_free_guidance: int = Input(
            description="Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.",
            default=3,
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
        seed: int = Input(
            description="Seed for random number generator. If `None` or `-1`, a random seed will be used.",
            default=None,
        ),
    ) -> Path:
        if text_chords == '':
             text_chords = None
        
        if text_chords and audio_chords and not continuation:
            raise ValueError("Must provide either only one of `audio_chords` or `text_chords`.")
        if text_chords and not bpm:
            raise ValueError("There must be `bpm` value set when text based chord conditioning.")
        if text_chords and (not time_sig or time_sig==""):
            raise ValueError("There must be `time_sig` value set when text based chord conditioning.")
        if continuation and not audio_chords:
            raise ValueError("Must provide an audio input file via `audio_chords` if continuation is `True`.")
        if multi_band_diffusion and int(self.model.lm.cfg.transformer_lm.n_q) == 8:
            raise ValueError("Multi-band Diffusion only works with non-stereo models.")
        
        if prompt is None:
             prompt = ''

        if time_sig is not None and not time_sig == '':
            if prompt == '':
                prompt = time_sig
            else:
                prompt = prompt + ', ' + time_sig
        if bpm is not None:
            if prompt == '':
                prompt = str(bpm)
            else:
                prompt = prompt + f', bpm : {bpm}'

        if model_version == "fine-tuned":
            try:
                self.model
            except AttributeError:
                raise Exception("ERROR: Fine-tuned weights don't exist! Is the model trained from `sakemin/musicgen-chord`? If not, set `model_version` from `chord`, `chord-large`, `stereo-chord` and `stereo-chord-large`.")
        else:
            if os.path.isfile(f'musicgen-{model_version}.th'):
                pass
            else:
                url = f"https://weights.replicate.delivery/default/musicgen-chord/musicgen-{model_version}.th"
                dest = f"musicgen-{model_version}.th"
                subprocess.check_call(["pget", url, dest], close_fds=False)
            self.model = load_ckpt(f'musicgen-{model_version}.th', self.device)
        self.model.lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        model = self.model

        set_generation_params = lambda duration: model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )

        model.lm.condition_provider.conditioners['self_wav'].chroma_coefficient = chroma_coefficient

        if not seed or seed == -1:
            seed = torch.seed() % 2 ** 32 - 1
            set_all_seeds(seed)
        set_all_seeds(seed)
        print(f"Using seed {seed}")

        '''
        if duration > 30:
            
            encodec_rate = 50
            sub_duration=25
            overlap = 30 - sub_duration
            wavs = []
            wav_sr = model.sample_rate
            set_generation_params(30)

            if (text_chords is None) and audio_chords is None: # Case 1
                wav, tokens = model.generate([prompt], progress=True, return_tokens=True)
                if multi_band_diffusion:
                    wav = self.mbd.tokens_to_wav(tokens)
                wavs.append(wav.detach().cpu())
                for i in range((duration - overlap) // sub_duration - 1):
                    wav, tokens= model.generate_continuation_with_audio_token(
                    prompt=tokens[...,sub_duration*encodec_rate:],
                    descriptions=[prompt],
                    progress=True,
                    return_tokens=True
                    )
                    if multi_band_diffusion:
                        wav = self.mbd.tokens_to_wav(tokens)
                    wavs.append(wav.detach().cpu())
                if (duration - overlap) % sub_duration != 0:
                    set_generation_params(overlap + ((duration - overlap) % sub_duration))
                    wav, tokens = model.generate_continuation_with_audio_token(
                        prompt=tokens[...,sub_duration*encodec_rate:],
                        descriptions=[prompt],
                        progress=True,
                        return_tokens=True
                    )
                    if multi_band_diffusion:
                        wav = self.mbd.tokens_to_wav(tokens)
                    wavs.append(wav.detach().cpu())
            elif (text_chords is None or text_chords == '') and audio_chords is not None: # Case 2
                audio_chords, sr = torchaudio.load(audio_chords)
                audio_chords = audio_chords[None] if audio_chords.dim() == 2 else audio_chords

                audio_start = 0 if not audio_start else audio_start
                if audio_end is None or audio_end == -1:
                    audio_end = audio_chords.shape[-1] / sr

                if audio_start > audio_end:
                    raise ValueError(
                        "`audio_start` must be less than or equal to `audio_end`"
                    )

                audio_chords = audio_chords[
                    ..., int(sr * audio_start) : int(sr * audio_end)
                ]
                wav, tokens = model.generate_with_chroma(['the intro of ' + prompt], audio_chords[...,:30*sr], sr, progress=True, return_tokens=True)
                if multi_band_diffusion:
                    wav = self.mbd.tokens_to_wav(tokens)
                wavs.append(wav.detach().cpu())
                for i in range(int((duration - overlap) // sub_duration) - 1):
                    wav, tokens = model.generate_continuation_with_audio_tokens_and_audio_chroma(
                    prompt=tokens[...,sub_duration*encodec_rate:],
                    melody_wavs = audio_chords[...,sub_duration*(i+1)*sr:(sub_duration*(i+1)+30)*sr],
                    melody_sample_rate=sr,
                    descriptions=['chorus of ' + prompt],
                    progress=True,
                    return_tokens=True
                    )
                    if multi_band_diffusion:
                        wav = self.mbd.tokens_to_wav(tokens)
                    wavs.append(wav.detach().cpu())
                if int(duration - overlap) % sub_duration != 0:
                    set_generation_params(overlap + ((duration - overlap) % sub_duration))
                    wav, tokens = model.generate_continuation_with_audio_tokens_and_audio_chroma(
                        prompt=tokens[...,sub_duration*encodec_rate:],
                        melody_wavs = audio_chords[...,sub_duration*(len(wavs))*sr:],
                        melody_sample_rate=sr,
                        descriptions=['the outro of ' + prompt],
                        progress=True,
                        return_tokens=True
                    )
                    if multi_band_diffusion:
                        wav = self.mbd.tokens_to_wav(tokens)
                    wavs.append(wav.detach().cpu())
            else: # Case 3
                wav, tokens = model.generate_with_text_chroma(descriptions = [prompt], chord_texts = [text_chords], bpm = [bpm], meter = [int(time_sig.split('/')[0])], progress=True, return_tokens=True)
                if multi_band_diffusion:
                    wav = self.mbd.tokens_to_wav(tokens)
                wavs.append(wav.detach().cpu())
                for i in range((duration - 10) // sub_duration - 1):
                    model.lm.condition_provider.conditioners['self_wav'].set_continuation_count(sub_duration/30, i)
                    wav, tokens = model.generate_continuation_with_audio_tokens_and_text_chroma(
                        tokens[...,sub_duration*encodec_rate:], [prompt], [text_chords], bpm=[bpm], meter=[int(time_sig.split('/')[0])], progress=True, return_tokens=True
                    )
                    if multi_band_diffusion:
                        wav = self.mbd.tokens_to_wav(tokens)
                    wavs.append(wav.detach().cpu())
                if (duration - overlap) % sub_duration != 0:
                    model.lm.condition_provider.conditioners['self_wav'].set_continuation_count(sub_duration/30, i+1)
                    set_generation_params(sub_duration + ((duration - overlap) % sub_duration))
                    wav, tokens = model.generate_continuation_with_audio_tokens_and_text_chroma(
                            tokens[...,sub_duration*encodec_rate:], [prompt], [text_chords], bpm=[bpm], meter=[int(time_sig.split('/')[0])], progress=True, return_tokens=True
                        )
                    if multi_band_diffusion:
                        wav = self.mbd.tokens_to_wav(tokens)
                    wavs.append(wav.detach().cpu())

            wav = wavs[0][...,:sub_duration*wav_sr]
            for i in range(len(wavs)-1):
                if i == len(wavs)-2:
                    wav = torch.concat([wav,wavs[i+1]],dim=-1)
                else:
                    wav = torch.concat([wav,wavs[i+1][...,:sub_duration*wav_sr]],dim=-1)

            wav = wav.cpu()
        else:
            '''
        if not audio_chords: 
            set_generation_params(duration)
            if text_chords is None or text_chords == '': # Case 4
                wav, tokens = model.generate([prompt], progress=True, return_tokens=True)
            else: # Case 5
                wav, tokens = model.generate_with_text_chroma(descriptions = [prompt], chord_texts = [text_chords], bpm = [bpm], meter = [int(time_sig.split('/')[0])], progress=True, return_tokens=True)
        else:
            audio_chords, sr = torchaudio.load(audio_chords)
            audio_chords = audio_chords[None] if audio_chords.dim() == 2 else audio_chords

            audio_start = 0 if not audio_start else audio_start
            if audio_end is None or audio_end == -1:
                audio_end = audio_chords.shape[2] / sr

            if audio_start > audio_end:
                raise ValueError(
                    "`audio_start` must be less than or equal to `audio_end`"
                )

            audio_chords_wavform = audio_chords[
                ..., int(sr * audio_start) : int(sr * audio_end)
            ]
            audio_chords_duration = audio_chords_wavform.shape[-1] / sr

            if continuation: 
                set_generation_params(duration)

                if text_chords is None or text_chords == '': # Case 6
                    wav, tokens  = model.generate_continuation(
                        prompt=audio_chords_wavform,
                        prompt_sample_rate=sr,
                        descriptions=[prompt],
                        progress=True,
                        return_tokens=True
                    )                        
                else: # Case 7
                    wav, tokens  = model.generate_continuation_with_text_chroma(
                        audio_chords_wavform, sr, [prompt], [text_chords], bpm=[bpm], meter=[int(time_sig.split('/')[0])], progress=True, return_tokens=True
                    )

            else: # Case 8
                set_generation_params(duration)
                wav, tokens = model.generate_with_chroma(
                    [prompt], audio_chords_wavform, sr, progress=True, return_tokens=True
                )

        if multi_band_diffusion:
            wav = self.mbd.tokens_to_wav(tokens)

        audio_write(
            "out",
            wav[0].cpu(),
            model.sample_rate,
            strategy=normalization_strategy,
        )
        wav_path = "out.wav"

        if output_format == "mp3":
            mp3_path = "out.mp3"
            if Path(mp3_path).exists():
                os.remove(mp3_path)
            subprocess.call(["ffmpeg", "-i", wav_path, mp3_path])
            os.remove(wav_path)
            path = mp3_path
        else:
            path = wav_path

        return Path(path)

    def _preprocess_audio(
        audio_path, model: MusicGen, duration: tp.Optional[int] = None
    ):

        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
        wav = wav.mean(dim=0, keepdim=True)

        # Calculate duration in seconds if not provided
        if duration is None:
            duration = wav.shape[1] / model.sample_rate

        # Check if duration is more than 30 seconds
        if duration > 30:
            raise ValueError("Duration cannot be more than 30 seconds")

        end_sample = int(model.sample_rate * duration)
        wav = wav[:, :end_sample]

        assert wav.shape[0] == 1
        assert wav.shape[1] == model.sample_rate * duration

        wav = wav.cuda()
        wav = wav.unsqueeze(1)

        with torch.no_grad():
            gen_audio = model.compression_model.encode(wav)

        codes, scale = gen_audio

        assert scale is None

        return codes


# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
