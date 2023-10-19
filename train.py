# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Entry point for dora to launch solvers for running training loops.
See more info on how to use dora: https://github.com/facebookresearch/dora
"""

import logging
import os
import os.path

import subprocess
from cog import BaseModel, Input, Path
import subprocess as sp
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from essentia.standard import (
    MonoLoader,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

class TrainingOutput(BaseModel):
    weights: Path

def prepare_data(
        dataset_path: Path,
        target_path: str = 'src/train_data',
        one_same_description: str = None,
        meta_path: str = 'src/meta',
        auto_labeling: bool = True,
        drop_vocals: bool = True):

    d_path = Path(target_path)
    d_path.mkdir(exist_ok=True, parents=True)

    # Decompressing file at dataset_path
    if str(dataset_path).rsplit('.', 1)[1] == 'zip':
        subprocess.run(['unzip', str(dataset_path), '-d', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'tar':
        subprocess.run(['tar', '-xvf', str(dataset_path), '-C', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'gz':
        subprocess.run(['tar', '-xvzf', str(dataset_path), '-C', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] == 'tgz':
        subprocess.run(['tar', '-xzvf', str(dataset_path), '-C', target_path + '/'])
    elif str(dataset_path).rsplit('.', 1)[1] in ['wav', 'mp3', 'flac']:
        import shutil
        shutil.move(str(dataset_path), target_path + '/' + str(dataset_path.name))
    else:
        raise Exception("Not supported compression file type. The file type should be one of 'zip', 'tar', 'tar.gz', 'tgz' types of compression file, or a single 'wav', 'mp3', 'flac' types of audio file.")

    # Removing __MACOSX and .DS_Store
    if (Path(target_path)/"__MACOSX").is_dir():
        import shutil
        shutil.rmtree(target_path+"/__MACOSX")
    elif (Path(target_path)/"__MACOSX").is_file():
        os.remove(target_path+"/__MACOSX")
    if (Path(target_path)/".DS_Store").is_dir():
        import shutil
        shutil.rmtree(target_path+"/.DS_Store")
    elif (Path(target_path)/".DS_Store").is_file():
        os.remove(target_path+"/.DS_Store")

    # Audio Chunking and Vocal Dropping
    from pydub import AudioSegment

    if drop_vocals:
        import demucs.api
        import torchaudio
        separator = demucs.api.Separator(model="mdx_extra")
    else:
        separator = None

    for filename in tqdm(os.listdir(target_path)):
        if filename.endswith(('.mp3', '.wav', '.flac')):
            if drop_vocals and separator is not None:
                print('Separating Vocals from ' + filename)
                origin, separated = separator.separate_audio_file(target_path + '/' + filename)
                mixed = separated["bass"] + separated["drums"] + separated["other"]
                torchaudio.save(target_path + '/' + filename, mixed, separator.samplerate)


            # Chuking audio files into 30sec chunks
            audio = AudioSegment.from_file(target_path + '/' + filename)

            audio = audio.set_frame_rate(44100) # Resampling to 44100
            
            if len(audio)>30000:
                print('Chunking ' + filename)

                # Splitting the audio files into 30-second chunks
                for i in range(0, len(audio), 30000):
                    chunk = audio[i:i + 30000]
                    if len(chunk) > 5000: # Omitting residuals with <5sec duration
                        chunk.export(f"{target_path + '/' + filename[:-4]}_chunk{i//1000}.wav", format="wav")
                os.remove(target_path + '/' + filename)

    max_sample_rate = 0
    import json

    # Auto Labeling
    if auto_labeling:
        sp.call(["curl", "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb", "--output", "genre_discogs400-discogs-effnet-1.pb"])
        sp.call(["curl", "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb", "--output", "discogs-effnet-bs64-1.pb"])
        sp.call(["curl", "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb", "--output", "mtg_jamendo_moodtheme-discogs-effnet-1.pb"])
        sp.call(["curl", "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb", "--output", "mtg_jamendo_instrument-discogs-effnet-1.pb"])


        from metadata import genre_labels, mood_theme_classes, instrument_classes
        import numpy as np

        def filter_predictions(predictions, class_list, threshold=0.1):
            predictions_mean = np.mean(predictions, axis=0)
            sorted_indices = np.argsort(predictions_mean)[::-1]
            filtered_indices = [i for i in sorted_indices if predictions_mean[i] > threshold]
            filtered_labels = [class_list[i] for i in filtered_indices]
            filtered_values = [predictions_mean[i] for i in filtered_indices]
            return filtered_labels, filtered_values

        def make_comma_separated_unique(tags):
            seen_tags = set()
            result = []
            for tag in ', '.join(tags).split(', '):
                if tag not in seen_tags:
                    result.append(tag)
                    seen_tags.add(tag)
            return ', '.join(result)

        def get_audio_features(audio_filename):
            audio = MonoLoader(filename=audio_filename, sampleRate=16000, resampleQuality=4)()
            embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
            embeddings = embedding_model(audio)

            result_dict = {}

            # Predicting genres
            genre_model = TensorflowPredict2D(graphFilename="genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
            predictions = genre_model(embeddings)
            filtered_labels, _ = filter_predictions(predictions, genre_labels)
            filtered_labels = ', '.join(filtered_labels).replace("---", ", ").split(', ')
            result_dict['genres'] = make_comma_separated_unique(filtered_labels)

            # Predicting mood/theme
            mood_model = TensorflowPredict2D(graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb")
            predictions = mood_model(embeddings)
            filtered_labels, _ = filter_predictions(predictions, mood_theme_classes, threshold=0.05)
            result_dict['moods'] = make_comma_separated_unique(filtered_labels)

            # Predicting instruments
            instrument_model = TensorflowPredict2D(graphFilename="mtg_jamendo_instrument-discogs-effnet-1.pb")
            predictions = instrument_model(embeddings)
            filtered_labels, _ = filter_predictions(predictions, instrument_classes)
            result_dict['instruments'] = filtered_labels

            return result_dict

        train_len = 0
        import librosa

        os.mkdir(meta_path)
        with open(meta_path + "/data.jsonl", "w") as train_file:
            files = list(d_path.rglob('*.mp3')) + list(d_path.rglob('*.wav'))
            if len(files)==0:
                raise ValueError("No audio file detected. Are you sure the audio file is longer than 5 seconds?")
            for filename in tqdm(files):
                # if filename.is_dir():
                    # continue

                result = get_audio_features(str(filename))

                # Obtaining key and BPM
                y, sr = librosa.load(str(filename))
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = round(tempo)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                key = np.argmax(np.sum(chroma, axis=1))
                key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key]
                length = librosa.get_duration(y=y, sr=sr)

                sr = librosa.get_samplerate(str(filename))
                if sr > max_sample_rate:
                    max_sample_rate = sr

                artist_name = ""

                entry = {
                    "key": f"{key}",
                    "artist": artist_name,
                    "sample_rate": sr,
                    "file_extension": "wav",
                    "description": "",
                    "keywords": "",
                    "duration": length,
                    "bpm": tempo,
                    "genre": result.get('genres', ""),
                    "title": "",
                    "name": "",
                    "instrument": result.get('instruments', ""),
                    "moods": result.get('moods', []),
                    "path": str(filename),
                }
                with open(str(filename).rsplit('.', 1)[0] + '.json', "w") as file:
                    json.dump(entry, file)
                print(entry)

                train_len += 1
                train_file.write(json.dumps(entry) + '\n')

            from numba import cuda
            device = cuda.get_current_device()
            device.reset()

            filelen = len(files)
    else:
        import audiocraft.data.audio_dataset

        meta = audiocraft.data.audio_dataset.find_audio_files(target_path, audiocraft.data.audio_dataset.DEFAULT_EXTS, progress=True, resolve=False, minimal=True, workers=10)

        if len(meta)==0:
            raise ValueError("No audio file detected. Are you sure the audio file is longer than 5 seconds?")
        
        for m in meta:
            if m.sample_rate > max_sample_rate:
                max_sample_rate = m.sample_rate
            fdict = {
                "key": "",
                "artist": "",
                "sample_rate": m.sample_rate,
                "file_extension": m.path.rsplit('.', 1)[1],
                "description": "",
                "keywords": "",
                "duration": m.duration,
                "bpm": "",
                "genre": "",
                "title": "",
                "name": Path(m.path).name.rsplit('.', 1)[0],
                "instrument": "",
                "moods": []
            }
            with open(m.path.rsplit('.', 1)[0] + '.json', "w") as file:
                json.dump(fdict, file)
        audiocraft.data.audio_dataset.save_audio_meta(meta_path + '/data.jsonl', meta)
        filelen = len(meta)

    audios = list(d_path.rglob('*.mp3')) + list(d_path.rglob('*.wav'))

    for audio in list(audios):
        jsonf = open(str(audio).rsplit('.', 1)[0] + '.json', 'r')
        fdict = json.load(jsonf)
        jsonf.close()

        # assert Path(str(audio).rsplit('.', 1)[0] + '.txt').exists() or Path(str(audio).rsplit('_chunk', 1)[0] + '.txt').exists() or one_same_description is not None

        if one_same_description is None:
            if Path(str(audio).rsplit('.', 1)[0] + '.txt').exists():
                f = open(str(audio).rsplit('.', 1)[0] + '.txt', 'r')
                line = f.readline()
                f.close()
                fdict["description"] = line
            elif Path(str(audio).rsplit('_chunk', 1)[0] + '.txt').exists():
                f = open(str(audio).rsplit('_chunk', 1)[0] + '.txt', 'r')
                line = f.readline()
                f.close()
                fdict["description"] = line
        else:
            fdict["description"] = one_same_description

        with open(str(audio).rsplit('.', 1)[0] + '.json', "w") as file:
            json.dump(fdict, file)

    return max_sample_rate, filelen

def train(
        dataset_path: Path = Input("Path to dataset directory. Input audio files will be chunked into multiple 30 second audio files. Must be one of 'tar', 'tar.gz', 'gz', 'zip' types of compressed file, or a single 'wav', 'mp3', 'flac' file. Audio files must be longer than 5 seconds.",),
        auto_labeling: bool = Input(description="Creating label data like genre, mood, theme, instrumentation, key, bpm for each track. Using `essentia-tensorflow` for music information retrieval.", default=True),
        drop_vocals: bool = Input(description="Dropping the vocal tracks from the audio files in dataset, by separating sources with Demucs.", default=True),
        one_same_description: str = Input(description="A description for all of audio data", default=None),
        epochs: int = Input(description="Number of epochs to train for", default=3),
        updates_per_epoch: int = Input(description="Number of iterations for one epoch", default=100),
        batch_size: int = Input(description="Batch size. Must be multiple of 8(number of gpus), for 8-gpu training.", default=16),
        optimizer: str = Input(description="Type of optimizer.", default='dadam', choices=["dadam", "adamw"]),
        lr: float = Input(description="Learning rate", default=1),
        lr_scheduler: str = Input(description="Type of lr_scheduler", default="cosine", choices=["exponential", "cosine", "polynomial_decay", "inverse_sqrt", "linear_warmup"]),
        warmup: int = Input(description="Warmup of lr_scheduler", default=0),
        cfg_p: float = Input(description="CFG dropout ratio", default=0.3),
) -> TrainingOutput:

    meta_path = 'src/meta'
    target_path = 'src/train_data'

    out_path = "trained_model.tar"

    # Removing previous training's leftover
    if os.path.isfile(out_path):
        os.remove(out_path)

    import shutil
    if os.path.isdir(meta_path):
        shutil.rmtree(meta_path)
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)
    if os.path.isdir('tmp'):
        shutil.rmtree('tmp')

    max_sample_rate, len_dataset = prepare_data(dataset_path, target_path, one_same_description, meta_path, auto_labeling, drop_vocals)

    if batch_size % 8 != 0:
        batch_size = batch_size - (batch_size%8)
        print(f"Batch size is reset to {batch_size}, the multiple of 8(number of gpus).")

    if batch_size > 16:
        batch_size = 16
        print(f"Batch size is reset to 16, since the maximum batch size for chord model with 8x A40 is 16.")


    solver = "musicgen/musicgen_chord_32khz"
    model_scale = "medium"
    conditioner = "chord2music"
    continue_from = "/src/musicgen_chord.th"

    args = ["run", "-d", "--", f"solver={solver}", f"model/lm/model_scale={model_scale}", f"continue_from={continue_from}", f"conditioner={conditioner}"]
    args.append(f"datasource.max_sample_rate={max_sample_rate}")
    args.append(f"datasource.train={meta_path}")
    args.append(f"dataset.train.num_samples={len_dataset}")
    args.append(f"optim.epochs={epochs}")
    args.append(f"optim.lr={lr}")
    args.append(f"schedule.lr_scheduler={lr_scheduler}")
    args.append(f"schedule.cosine.warmup={warmup}")
    args.append(f"schedule.polynomial_decay.warmup={warmup}")
    args.append(f"schedule.inverse_sqrt.warmup={warmup}")
    args.append(f"schedule.linear_warmup.warmup={warmup}")
    args.append(f"classifier_free_guidance.training_dropout={cfg_p}")
    if updates_per_epoch is not None:
        args.append(f"logging.log_updates={updates_per_epoch//10 if updates_per_epoch//10 >=1 else 1}")
    else:
        args.append(f"logging.log_updates=0")
    args.append(f"dataset.batch_size={batch_size}")
    args.append(f"optim.optimizer={optimizer}")

    if updates_per_epoch is None:
        args.append("dataset.train.permutation_on_files=False")
        args.append("optim.updates_per_epoch=1")
    else:
        args.append("dataset.train.permutation_on_files=True")
        args.append(f"optim.updates_per_epoch={updates_per_epoch}")

    print(os.getcwd())
    sp.call(["dora"]+args)

    for dirpath, dirnames, filenames in os.walk("tmp"):
        for filename in [f for f in filenames if f == "checkpoint.th"]:
            checkpoint_dir = os.path.join(dirpath, filename)

    loaded = torch.load(checkpoint_dir, map_location=torch.device('cpu'))

    torch.save({'xp.cfg': loaded["xp.cfg"], "model": loaded["model"]}, out_path)

    return TrainingOutput(weights=Path(out_path))
