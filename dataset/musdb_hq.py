from torchaudio.datasets import MUSDB_HQ


if __name__ == "__main__":
    # Download the dataset
    musdb = MUSDB_HQ(root="dataset", subset="train", download=True)

    # Access a track
    track = musdb[0]
    piece = track.audio
    # Play the mixture
    track.audio

    # # Play the drums
    # track.targets["drums"].audio

    # # Play the bass
    # track.targets["bass"].audio

    # # Play the vocals
    # track.targets["vocals"].audio

    # # Play the other
    # track.targets["other"].audio

    # # Play the mixture
    # track.audio

    # # Play the drums
    # track.targets["drums"].audio

    # # Play the bass
    # track.targets["bass"].audio

    # # Play the vocals
    # track.targets["vocals"]