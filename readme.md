# Audio Separation Project

An assignment for artificial intelligence department. A simple implementation of a music separation project, which refer to this repository: [Demucs](https://github.com/facebookresearch/demucs)


## 1 Direct Implementation

### 1.0 OS
- [X] Windows10
- [X] Ubuntu20.04
- [X] macOS (CPU only)

### 1.1 modify the configs/config.yaml

### 1.2 type in terminal:
```
python run.py --c configs/config.yaml
```

And then you will see results from `records` folder.



## 2 Train model

### 2.1 Dataset

The first thing to do should always be data. We use the following dataset from [zenodo](https://zenodo.org/), an open source website that holds all kinds of data. We select the following one:

- [MUSDB18](https://zenodo.org/records/1117372)

For Southeast University student, we upload the dataset to `pan.seu.edu.cn` to fasten your downloading. Here is a link:




After downloading and unzip, please change its format from `mp3` into `.wav`, since the current(Nov 2023) torchaudio only support `wav` format. You can directly run the following bash(**remember to change the location!**), here I recommend you to put the musicdb18 into a parallel position with the project:
```
audioSep Project
|--changeAudioFormat.bash
|....
musicdb18
|-- piece1.mp3
|-- piece1.mp3
|...
```

```terminal
# please ensure you are at the current project work space
chmod +x changeAudioFormat.bash
./changeAudioFormat.bash
```

After running, you will see folder `musicdb18_wav` in your project folder. For more detailed information about this dataset, please refer to the [introduction site](https://zenodo.org/records/1117372) or click the readme under downloaded original dataset folder.




## 3 Some 

### 3.1 About metrics used

#### STOI(Short-Time Objective Intelligibility)

Attention! it could only be used to evaluate **mono channel audio**.

The stoi function is designed to evaluate the intelligibility of speech signals, which are typically mono. Intelligibility is a measure of how comprehensible speech is in given conditions, and for this measurement, stereo or multi-channel audio does not provide additional information compared to mono audio.

If the source or predicted audio is stereo (i.e., has 2 channels), it's common practice to either:
- Average the channels to get a mono signal.
- [X] (The one we adopt)Evaluate the metric on each channel separately and then average the results.


