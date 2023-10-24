from torchaudio.transforms import Fade
from model.model import HDemucs
import torch, torchaudio
import os

def get_model(args) -> torch.nn.Module:
    """Construct the model and load the pretrained weight."""
    model = HDemucs(sources=args.sources, nfft=args.nfft, depth=args.depth)
    if os.path.isfile(args.checkpoint_path):
        state_dict = torch.load(args.checkpoint_path)
        model.load_state_dict(state_dict)
    else:
        path = torchaudio.utils.download_asset(args.download_path)
        if args.use_gpu:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        save_model(args, model)
    model.eval()
    return model

def save_model(args, model):
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = model_to_save.state_dict()
    torch.save(checkpoint, args.save_dir + "backup/model.pt")

def separate_sources(
    args,
    model,
    mix,
    segment=10.0,
    overlap=0.1,
    device=None,
):
    device = torch.device(args.device_gpu if torch.cuda.is_available() and args.use_gpu else "cpu")
    batch, channels, length = mix.shape

    chunk_len = int(args.sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * args.sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


