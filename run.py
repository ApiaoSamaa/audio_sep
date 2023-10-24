import torch
import torchaudio
from torchaudio.datasets import MUSDB_HQ
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

def output_format(args, original_source: torch.Tensor, predicted_source: torch.Tensor, source: str, stft, sample_rate):
    print(
        "SDR score is:",
        separation.bss_eval_sources(original_source.detach().numpy(), predicted_source.detach().numpy())[0].mean(),
    )
    save_spectrum(args=args, stft=stft(predicted_source)[0], title=f"Spectrogram-{source}")
    return Audio(predicted_source, rate=sample_rate)


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
    SAMPLE_SONG = download_asset("tutorial-assets/hdemucs_mix.wav")
    waveform, sample_rate = torchaudio.load(SAMPLE_SONG)  # replace SAMPLE_SONG with desired path for different song
    waveform = waveform.to(device)
    mixture = waveform
    tlogger.print()

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
    drums_original = download_asset("tutorial-assets/hdemucs_drums_segment.wav")
    bass_original = download_asset("tutorial-assets/hdemucs_bass_segment.wav")
    vocals_original = download_asset("tutorial-assets/hdemucs_vocals_segment.wav")
    other_original = download_asset("tutorial-assets/hdemucs_other_segment.wav")
    tlogger.print()
    
    
    drums_spec = audios["drums"][:, frame_start:frame_end].cpu()
    drums, sample_rate = torchaudio.load(drums_original)

    bass_spec = audios["bass"][:, frame_start:frame_end].cpu()
    bass, sample_rate = torchaudio.load(bass_original)

    vocals_spec = audios["vocals"][:, frame_start:frame_end].cpu()
    vocals, sample_rate = torchaudio.load(vocals_original)

    other_spec = audios["other"][:, frame_start:frame_end].cpu()
    other, sample_rate = torchaudio.load(other_original)

    mix_spec = mixture[:, frame_start:frame_end].cpu()
    # Mixture Clip
    save_spectrum(args=args, stft = stft(mix_spec)[0], title = "Spectrogram-Mixture")
    Audio(mix_spec, rate=sample_rate)
    
    output_results = partial(output_format,args=args, stft=stft, sample_rate=sample_rate)
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

