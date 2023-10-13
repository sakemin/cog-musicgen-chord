# Cog Implementation of MusicGen-Chord
[![Replicate](https://replicate.com/sakemin/musicgen-chord/badge)](https://replicate.com/sakemin/musicgen-chord) 

MusicGen Chord is the modified version of Meta's [MusicGen](https://github.com/facebookresearch/audiocraft) Melody model, which can generate music based on audio-based chord conditions or text-based chord conditions.

You can demo this model or learn how to use it with Replicate's API [here](https://replicate.com/sakemin/musicgen-chord). 

# Run with Cog

[Cog](https://github.com/replicate/cog) is an open-source tool that packages machine learning models in a standard, production-ready container. 
You can deploy your packaged model to your own infrastructure, or to [Replicate](https://replicate.com/), where users can interact with it via web interface or API.

## Prerequisites 

**Cog.** Follow these [instructions](https://github.com/replicate/cog#install) to install Cog, or just run: 

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

Note, to use Cog, you'll also need an installation of [Docker](https://docs.docker.com/get-docker/).

* **GPU machine.** You'll need a Linux machine with an NVIDIA GPU attached and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. If you don't already have access to a machine with a GPU, check out our [guide to getting a 
GPU machine](https://replicate.com/docs/guides/get-a-gpu-machine).

## Step 1. Clone this repository

```sh
git clone https://github.com/sakemin/cog-musicgen-chord
```

## Step 2. Run the model

To run the model, you need a local copy of the model's Docker image. You can satisfy this requirement by specifying the image ID in your call to `predict` like:

```
cog predict r8.im/sakemin/musicgen-chord@sha256:f60d0c00ed7bf642f639224123c456635fe203470d6f6e80545aaa405ad1252a -i prompt="k pop, cool synthwave, drum and bass with jersey club beats" -i duration=30 -i text_chords="C G A:min F" -i bpm=140 -i time_sig="4/4"
```

For more information, see the Cog section [here](https://replicate.com/sakemin/musicgen-chord/api#run)

Alternatively, you can build the image yourself, either by running `cog build` or by letting `cog predict` trigger the build process implicitly. For example, the following will trigger the build process and then execute prediction: 

```
cog predict -i prompt="k pop, cool synthwave, drum and bass with jersey club beats" -i duration=30 -i text_chords="C G A:min F" -i bpm=140 -i time_sig="4/4"
```

Note, the first time you run `cog predict`, model weights and other requisite assets will be downloaded if they're not available locally. This download only needs to be executed once.

# Run on replicate

## Step 1. Ensure that all assets are available locally

If you haven't already, you should ensure that your model runs locally with `cog predict`. This will guarantee that all assets are accessible. E.g., run: 

```
cog predict -i prompt="k pop, cool synthwave, drum and bass with jersey club beats" -i duration=30 -i text_chords="C G A:min F" -i bpm=140 -i time_sig="4/4"
```

## Step 2. Create a model on Replicate.

Go to [replicate.com/create](https://replicate.com/create) to create a Replicate model. If you want to keep the model private, make sure to specify "private".

## Step 3. Configure the model's hardware

Replicate supports running models on variety of CPU and GPU configurations. For the best performance, you'll want to run this model on an A100 instance.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

## Step 4: Push the model to Replicate


Log in to Replicate:

```
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 1:

```
cog push r8.im/username/modelname
```

[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)
---
## Prediction
### Prediction Parameters
- `prompt` (`string`) : A description of the music you want to generate.
- `text_chords` (`string`) : A text based chord progression condition. Single uppercase alphabet character(eg. `C`) is considered as a major chord. Chord attributes like(`maj`, `min`, `dim`, `aug`, `min6`, `maj6`, `min7`, `minmaj7`, `maj7`, `7`, `dim7`, `hdim7`, `sus2` and `sus4`) can be added to the root alphabet character after `:`.(eg. `A:min7`) Each chord token splitted by `SPACE` is allocated to a single bar. If more than one chord must be allocated to a single bar, cluster the chords adding with `,` without any `SPACE`.(eg. `C,C:7 G, E:min A:min`) You must choose either only one of `audio_chords` below or `text_chords`.
- `bpm` (`number`) : BPM condition for the generated output. `text_chords` will be processed based on this value. This will be appended at the end of `prompt`.
- `time_sig` (`string`) : Time signature value for the generate output. `text_chords` will be processed based on this value. This will be appended at the end of `prompt`.
- `audio_chord` (`file`) : An audio file that will condition the chord progression. You must choose only one among `audio_chords` or `text_chords` above.
- `audio_start` (`integer`) : Start time of the audio file to use for chord conditioning.(Default = 0)
- `audio_end` (`integer`) : End time of the audio file to use for chord conditioning. If None, will default to the end of the audio clip.
- `duration` (`integer`) : Duration of the generated audio in seconds.(Default = 8)
- `continuation` (`boolean`) : If `True`, generated music will continue from `audio_chords`. If chord conditioning, this is only possible when the chord condition is given with `text_chords`. If `False`, generated music will mimic `audio_chords`'s chord.
- `multi_band_diffusion` (`boolean`) : If `True`, the EnCodec tokens will be decoded with MultiBand Diffusion.
- `normalization_strategy` (`string`) : Strategy for normalizing audio.(Allowed values : `loudness`, `clip`, `peak`, `rms` / Default value = `loudness`)
- `top_k` (`integer`) : Reduces sampling to the k most likely tokens.(Default = 250)
- `top_p` (`number`) : Reduces sampling to tokens with cumulative probability of p. When set to `0` (default), top_k sampling is used.(Default = 0)
- `temperature` (`number`) : Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.(Default = 1)
- `classifier_free_guidance` (`integer`) : Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.(Default = 3)
- `output_format` (`string`) : Output format for generated audio.(Allowed values : `wav`, `mp3` / Default = `wav`)
- `seed` (`integer`) : Seed for random number generator. If `None` or `-1`, a random seed will be used.
  
## Text Based Chord Conditioning
### Text Chord Condition Format
- `SPACE` is used as split token. Each splitted chunk is assigned to a single bar.
	-	`C G E:min A:min`
- When multiple chords must be assigned in a single bar, then append more chords with `,`.
	-	`C G,G:7 E:min,E:min7 A:min`
- Chord type can be specified after `:`.
	- 	Just using a single uppercase alphabet(eg. `C`, `E`) is considered as a major chord.
	-	 `maj`, `min`, `dim`, `aug`, `min6`, `maj6`, `min7`, `minmaj7`, `maj7`, `7`, `dim7`, `hdim7`, `sus2` and `sus4` can be appended with `:`.
		- 	eg. `E:dim`, `B:sus2`
- 'sharp' and 'flat' can be specified with `#` and `b`.
	- 	eg. `E#:min` `Db`
### BPM and Time Signature
- To create chord chroma, `bpm` and `time_sig` values must be specified.
	- `bpm` can be a float value. (eg. `132`, `60`)
	- The format of `time_sig` is `(int)/(int)`. (eg. `4/4`, `3/4`, `6/8`, `7/8`, `5/4`)
- `bpm` and `time_sig` values will be automatically concatenated after `prompt` description value, so you don't need to specify bpm or time signature information in the description for `prompt`.

## Audio Based Chord Conditioning
### Audio Chord Conditioning Instruction
- You can also give chord condition with `audio_chords`.
- With `audio_start` and `audio_end` values, you can specify which part of the `audio_chords` file input will be used as chord condition.
- The chords will be recognized from the `audio_chords`, using [BTC](https://github.com/jayg996/BTC-ISMIR19) model. 

## Additional Feature
### Continuation
- If `continuation` is `True`, then the input audio file given at `audio_chords` will not be used as audio chord condition. The generated music output will be continued from the given file.
- You can also use `audio_start` and `audio_end` values to crop the input audio file.
### Infinite Generation
- You can set `duration` longer than 30 seconds.
- Due to MusicGen's limitation of generating a maximum 30-second audio in one iteration, if the specified duration exceeds 30 seconds, the model will create multiple sequences. It will utilize the latter portion of the output from the previous generation step as the audio prompt (following the same continuation method) for the subsequent generation step.
### Multi-Band Diffusion
- [Multi-Band Diffusion(MBD)](https://github.com/facebookresearch/audiocraft/blob/main/docs/MBD.md) is used for decoding the EnCodec tokens.
- If the tokens are decoded with MBD, than the output audio quality is better.
- Using MBD takes more calculation time, since it has its own prediction sequence.
---
## Licenses
- All code in this repository is licensed under the [Apache License 2.0 license](https://github.com/sakemin/cog-musicgen-chord/blob/main/LICENSE).
- The weights in [this repository](https://github.com/sakemin/cog-musicgen-chord) repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights file](https://github.com/sakemin/cog-musicgen-chord/blob/main/LICENSE_weights).
- The code in the [Audiocraft](https://github.com/facebookresearch/audiocraft) repository is released under the MIT license (see [LICENSE file](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE)).
- The weights in the [Audiocraft](https://github.com/facebookresearch/audiocraft) repository are released under the CC-BY-NC 4.0 license (see [LICENSE_weights file](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE_weights)).
