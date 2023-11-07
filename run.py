import torch
import torchaudio
from functools import partial
import matplotlib.pyplot as plt
from IPython.display import Audio
from mir_eval import separation
from torchaudio.utils import download_asset
from utils.config_utils import get_args, load_yaml, build_record_folder
from utils.costom_logger import timeLogger
from utils.model_util import get_model
from utils.model_util import separate_sources
import os
from pystoi import stoi
from pesq import pesq
import pandas as pd
from openpyxl import load_workbook, Workbook
from tqdm import tqdm
import csv


def compute_pesq(rate, original, predicted):
    # Handle multi-channel by averaging PESQ over channels
    # Downsample to 16kHz
    new_freq = 16000
    resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
    original_downsampled = resampler(torch.tensor(original))
    predicted_downsampled = resampler(torch.tensor(predicted))
    
    try:
        if original.ndim > 1:
            channel_pesqs = [pesq(new_freq, original_downsampled[i].numpy(), predicted_downsampled[i].numpy(), 'nb') for i in range(original_downsampled.shape[0])]
            return sum(channel_pesqs) / len(channel_pesqs)
        else:
            return pesq(rate, original, predicted, 'nb')
    except Exception as e:
        print(f"PESQ zero: {e}")
        return 0.0
    
def compute_stoi(rate, original, predicted):
    try:
        if original.ndim > 1:
            channel_stoi = [stoi(original[i].detach().numpy(), predicted[i].detach().numpy(), rate, extended=False) for i in range(original.shape[0])]
            return sum(channel_stoi) / len(channel_stoi)
        else:
            return stoi(original.detach().numpy(), predicted.detach().numpy(), rate, extended=False)
    except Exception as e:
        print(f"STOI Zero: {e}")
        return 0.0

from mir_eval import separation

def compute_metrics(original, predicted, sample_rate):
    # Compute SDR
    try:
        sdr = separation.bss_eval_sources(original.detach().numpy(), predicted.detach().numpy())[0].mean()
    except Exception as e:
        print(f"SDR Zero: {e}")
        sdr = 0.0
    # Compute STOI
    try:
        st = compute_stoi(sample_rate, original, predicted)
    except Exception as e:
        print(f"STOI Zero: {e}")
        st = 0.0
    # Compute PESQ
    pesq_score = compute_pesq(sample_rate, original.detach().numpy(), predicted.detach().numpy())
    
    # Compute SNR
    noise = original - predicted
    snr = 10 * torch.log10(torch.mean(original**2) / torch.mean(noise**2))
    try:
        sir = separation.bss_eval_sources(original.detach().numpy(), predicted.detach().numpy())[1].mean()
    except Exception as e:
        print(f"SIR Zero: {e}")
        sir = 0.0
    return sdr, st, pesq_score, snr, sir

def save_spectrum(args, stft, title="Spectrogram"):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    _, axis = plt.subplots(1, 1)
    axis.imshow(spectrogram, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto")
    axis.set_title(title)
    plt.tight_layout()
    args.img_dir = args.save_dir + "backup/imgs/"
    os.makedirs(args.img_dir, exist_ok=True)
    plt.savefig(args.img_dir+title+".png", bbox_inches='tight')  # Save the figure to the provided path
    plt.close()  # Close the figure to free up memory

def save_audio(args, audio, title="Audio"):
    args.audio_dir = args.save_dir + "backup/audios/"
    os.makedirs(args.audio_dir, exist_ok=True)
    torchaudio.save(args.audio_dir+title+".wav", audio, sample_rate=44100)

"""
data= {
    'SDR': sdr,
    'STOI': st,
    'PESQ': pe,
    'SNR': sn.item(),
    'SIR': sir
}
"""
def save_data_to_csv(data, song_name, source, csv_filename):
    with open(csv_filename, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([song_name, source, data['SDR'], data['STOI'], data['PESQ'], data['SNR'], data['SIR']])
            
def output_format(args, original_source: torch.Tensor, predicted_source: torch.Tensor, source: str, stft, sample_rate,song_path):
    sdr, st, pe, sn, sir = compute_metrics(original_source, predicted_source, sample_rate)
    song_name = os.path.basename(song_path).split('.stem_')[0]
    
    msg = f"[Saved Results for {args.project_name},{args.exp_name}]\n"
    msg = f"=========={song_name}:{source}===========\n"
    msg += f"SDR score for {source}: {sdr}\n"
    msg += f"STOI score for {source}: {st}\n"
    msg += f"PESQ score for {source}: {pe}\n"
    msg += f"SNR score for {source}: {sn.item()}\n"
    msg += f"SIR score for {source}: {sir}\n"
    
    print(f"=========={song_name}:{source}===========\n")
    print(f"SDR score for {source}:", sdr)
    print(f"STOI score for {source}:", st)
    print(f"PESQ score for {source}:", pe)
    print(f"SNR score for {source}:", sn.item())
    print(f"SIR score for {source}:", sir)
    
    with open(args.save_dir + "logs.txt", "a+") as ftxt:
        ftxt.write(msg)
    csv_filename = args.save_dir+'results.csv'
    data= {
        'SDR': sdr,
        'STOI': st,
        'PESQ': pe,
        'SNR': sn.item(),
        'SIR': sir
    }
    
    save_data_to_csv(data, song_name, source, csv_filename)
    save_spectrum(args=args, stft=stft(predicted_source)[0], title=f"Predicted_Spectrogram-{source}")
    save_spectrum(args=args, stft=stft(original_source)[0], title=f"Original_Spectrogram-{source}")
    # save_audio(args=args, audio=predicted_source, title=f"Predicted_Audio-{source}")
    # save_audio(args=args, audio=original_source, title=f"Original_Audio-{source}")
    return Audio(predicted_source, rate=sample_rate)


# def test(args, model, device, sample_rate, song_path):
    
def song_filepaths(directory):
    """Return a list of absolute file paths directly under the given directory."""
    return [os.path.abspath(os.path.join(directory, file))
            for file in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file))]



def main(args, tlogger):
    tlogger.print("Loading Model...")
    model = get_model(args)
    tlogger.print()
    
    device = torch.device(args.device_gpu if torch.cuda.is_available() and args.use_gpu else "cpu")
    model.to(device)
    sample_rate = args.sample_rate

    print(f"Sample rate: {sample_rate}")

    # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
    tlogger.print("Loading Audio...")
    all_songs = song_filepaths(args.test_dir)
    tlogger.print()
    
    csv_filename = args.save_dir+'results.csv'
    f = ['song_name', 'source', 'SDR', 'STOI', 'PESQ', 'SNR', 'SIR']
    with open(csv_filename, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(f)
    for song_mixture in tqdm(all_songs, desc="begin testing"):
        print("\n")
        print(song_mixture)
        # SAMPLE_SONG = "./data/DSD100subset/Mixtures/Dev/055_I_am_Alright/mixture_5s.wav"
        waveform, sample_rate = torchaudio.load(song_mixture)  # replace SAMPLE_SONG with desired path for different song
        waveform = waveform.to(device)
        mixture = waveform
        

        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()  # normalization
        tlogger.print("Separating track...")
        sources = separate_sources(
            args=args,
            model=model,
            mix=waveform[None],
            device=device,
            segment=args.segment,
            overlap=args.overlap,
        )[0]
        tlogger.print()
        
        sources = sources * ref.std() + ref.mean()

        sources_list = model.sources
        sources = list(sources)

        audios = dict(zip(sources_list, sources))

        stft = torchaudio.transforms.Spectrogram(
            n_fft=args.nfft,
            hop_length=args.nhop,
            power=None,
        )

        frame_start = args.segment_start * sample_rate
        frame_end = args.segment_end * sample_rate
        tlogger.print("Loading ground truth separation...")
        song_original_prefix = './musdb18_wav/test/Sources/'+song_mixture.split('/')[-1].split('.stem_')[0]+'.'
        
        
        drums_original = song_original_prefix + 'stem_drums.wav'
        bass_original = song_original_prefix + 'stem_bass.wav'
        vocals_original = song_original_prefix + 'stem_vocals.wav'
        other_original = song_original_prefix + 'stem_accompaniment.wav'
        tlogger.print()

        drums_spec = audios["drums"][:, frame_start:frame_end].cpu()
        drums, sample_rate = torchaudio.load(drums_original)
        drums = drums[:, frame_start:frame_end].cpu()
        # breakpoint()
        # drums, sample_rate = torchaudio.load(drums_original)
        bass_spec = audios["bass"][:, frame_start:frame_end].cpu()
        bass, sample_rate = torchaudio.load(bass_original)
        bass = bass[:, frame_start:frame_end].cpu()
        
        vocals_spec = audios["vocals"][:, frame_start:frame_end].cpu()
        vocals, sample_rate = torchaudio.load(vocals_original)
        vocals = vocals[:, frame_start:frame_end].cpu()
        
        other_spec = audios["other"][:, frame_start:frame_end].cpu()
        other, sample_rate = torchaudio.load(other_original)
        other = other[:, frame_start:frame_end].cpu()

        mix_spec = mixture[:, frame_start:frame_end].cpu()
        # Mixture Clip
        save_spectrum(args=args, stft = stft(mix_spec)[0], title = "Spectrogram-Mixture")
        Audio(mix_spec, rate=sample_rate)
        
        output_results = partial(output_format,args=args, stft=stft, sample_rate=sample_rate, song_path = song_mixture)
        # Drums Clip
        output_results(original_source=drums, predicted_source=drums_spec, source="drums")
        # Bass Clip
        output_results(original_source=bass, predicted_source=bass_spec, source="bass")
        # Vocals Audio
        output_results(original_source=vocals, predicted_source=vocals_spec, source="vocals")
        # Other Clip
        output_results(original_source=other, predicted_source=other_spec, source="other")


if __name__ == "__main__":
    tlogger = timeLogger()
    tlogger.print("Reading Config...")
    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    build_record_folder(args)
    tlogger.print()
    main(args, tlogger)

