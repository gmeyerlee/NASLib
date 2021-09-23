import pathlib

import torch
import torchaudio
import torchvision
import numpy as np
from .utils import get_project_root

torchaudio.set_audio_backend('sox_io')

def split_with_pad(line, sep, min_len, pad_with=''):
    parts = line.split(sep)
    if len(parts) < min_len:
        missing = (min_len - len(parts))
        parts.extend([pad_with] * missing)
    return parts


def get_transforms(part_name):
    transforms = torchvision.transforms.Compose([
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, win_length=400, hop_length=160, n_mels=80),
        torch.log,
        get_normalize_fn(part_name)
    ])

    return transforms

def collate_fn(batch):
    import ipdb; ipdb.set_trace()
    audio = [b[0][0] for b in batch]
    audio_lengths = [a.shape[-1] for a in audio]
    sentence = [b[1] for b in batch]
    sentence_lengths = [len(s) for s in sentence]
    audio = pad_sequence_bft(audio, extra=0, padding_value=0.0)
    sentence = pad_sentences(sentence, padding_value=0.0)
    return (audio, torch.tensor(audio_lengths)), (torch.tensor(sentence, dtype=torch.int32), torch.tensor(sentence_lengths))



class PhonemeEncoder:
    all_encodings = [61, 48, 39]

    def __init__(self, num_classes, remove_folded=True):
        assert num_classes in PhonemeEncoder.all_encodings

        self.num_classes = num_classes
        self.class_idx = PhonemeEncoder.all_encodings.index(self.num_classes)
        self.remove_folded = remove_folded

        num_encodings = len(PhonemeEncoder.all_encodings)

        mapping = pathlib.Path(get_project_root()).joinpath('data/timit/timit_folding.txt').read_text().strip().split('\n')
        mapping = [split_with_pad(line, '\t', num_encodings) for line in mapping]
        self.mappings = {}
        self.to_delete = {}
        for src in range(num_encodings-1):
            self.mappings[src] = {}
            self.to_delete[src] = {}
            for dst in range(src+1, num_encodings):
                self.mappings[src][dst] = { line[src]: line[dst] for line in mapping }
                self.to_delete[src][dst] = set(p for p, v in self.mappings[src][dst].items() if not v)

        self.encodeds = [set(line[idx] for line in mapping if line[idx]) for idx in range(num_encodings)]
        self.encodeds = [sorted(list(encodeds)) for encodeds in self.encodeds]

        self.idx_mappings = {}
        for src in range(num_encodings-1):
            self.idx_mappings[src] = {}
            for dst in range(src+1, num_encodings):
                self.idx_mappings[src][dst] = { 0: 0 }
                for src_idx, src_ph in enumerate(self.encodeds[src]):
                    dst_ph = self.mappings[src][dst][src_ph]
                    dst_idx = self.encodeds[dst].index(dst_ph)+1 if dst_ph else 0
                    self.idx_mappings[src][dst][src_idx+1] = dst_idx

    def get_vocab(self, inc_blank=False, num_classes=None):
        class_idx = PhonemeEncoder.all_encodings.index(num_classes) if num_classes is not None else self.class_idx
        ret = list(self.encodeds[class_idx])
        if inc_blank:
            ret = ['_'] + ret
        return ret

    def _fold(self, phonemes, dst_class_idx=None):
        if dst_class_idx is None:
            dst_class_idx = self.class_idx
        if dst_class_idx == 0:
            return phonemes

        return [self.mappings[0][dst_class_idx][p] for p in phonemes if not self.remove_folded or p not in self.to_delete[0][dst_class_idx]]

    def fold_encoded(self, encodeds, num_classes):
        if num_classes >= self.num_classes:
            return encodeds
        if num_classes not in PhonemeEncoder.all_encodings:
            raise ValueError(num_classes)

        new_class_idx = PhonemeEncoder.all_encodings.index(num_classes)
        for old_idx, new_idx in self.idx_mappings[self.class_idx][new_class_idx].items():
            encodeds[encodeds == old_idx] = new_idx

        return encodeds


    def encode(self, phonemes):
        phonemes_folded = self._fold(phonemes)
        enc = [self.encodeds[self.class_idx].index(p)+1 if p else 0 for p in phonemes_folded] #start from 1, 0 is used for blank
        return enc

    def decode(self, encodeds):
        return [self.encodeds[self.class_idx][idx-1] if idx else '' for idx in encodeds]


class TimitDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder, encoder, subset='TRAIN', ignore_sa=True, transforms=None):
        root = pathlib.Path(root_folder).expanduser() / 'data'
        wavs = list(root.rglob(f'{subset}/**/*.WAV.wav'))
        # wavs = list(root.rglob(f'{subset}/**/*.RIFF.WAV'))
        wavs = sorted(wavs)
        if ignore_sa:
            wavs = [w for w in wavs if not w.name.startswith('SA')]
        phonemes = [(f.parent / f.stem).with_suffix('.PHN') for f in wavs]

        self.audio = []
        self.audio_len = []
        for wav in wavs:
            tensor, sample_rate = torchaudio.load(str(wav))
            self.audio.append(tensor)
            self.audio_len.append(tensor.shape[1] / sample_rate)

        def load_sentence(f):
            lines = f.read_text().strip().split('\n')
            last = [l.rsplit(' ', maxsplit=1)[-1] for l in lines]
            last = encoder.encode(last)
            return last

        self.root_folder = root_folder
        self.encoder = encoder
        self.sentences = [load_sentence(f) for f in phonemes]
        self.transforms = transforms

        assert len(self.audio) == len(self.sentences)

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        audio = self.audio[idx]
        sentence = self.sentences[idx]
        if self.transforms is not None:
            audio = self.transforms(audio)
        return audio, sentence

    def get_indices_shorter_than(self, time_limit):
        return [i for i, audio_len in enumerate(self.audio_len) if time_limit is None or audio_len < time_limit]


def pad_sequence_bft(sequences, extra=0, padding_value=0.0):
    batch_size = len(sequences)
    leading_dims = sequences[0].shape[:-1]
    max_t = max([s.shape[-1]+extra for s in sequences])

    out_dims = (batch_size, ) + leading_dims + (max_t, )

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[-1]
        out_tensor[i, ..., :length] = tensor

    return out_tensor


def pad_sentences(sequences, padding_value=0.0):
    max_t = max([len(s) for s in sequences])
    sequences = [s+[0]*(max_t-len(s)) for s in sequences]
    return sequences


def get_normalize_fn(part_name, eps=0.001):
    stats = np.load(pathlib.Path(get_project_root()).joinpath(f'data/timit/timit_train_stats.npz'))
    mean = stats['moving_mean'][None,:,None]
    variance = stats['moving_variance'][None,:,None]
    def normalize(audio):
        return (audio - mean) / (variance + eps)
    return normalize


def get_dataloaders(timit_root, batch_size):
    encoder = PhonemeEncoder(48)

    def get_transforms(part_name):
        transforms = torchvision.transforms.Compose([
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, win_length=400, hop_length=160, n_mels=80),
            torch.log,
            get_normalize_fn(part_name)
        ])

        return transforms

    def collate_fn(batch):
        audio = [b[0][0] for b in batch]
        audio_lengths = [a.shape[-1] for a in audio]
        sentence = [b[1] for b in batch]
        sentence_lengths = [len(s) for s in sentence]
        audio = pad_sequence_bft(audio, extra=0, padding_value=0.0)
        sentence = pad_sentences(sentence, padding_value=0.0)
        return (audio, torch.tensor(audio_lengths)), (torch.tensor(sentence, dtype=torch.int32), torch.tensor(sentence_lengths))


    subsets = ['TRAIN', 'VAL', 'TEST']
    datasets = [TimitDataset(timit_root, encoder, subset=s, ignore_sa=True, transforms=get_transforms(s)) for s in subsets]
    train_sampler = torch.utils.data.SubsetRandomSampler(datasets[0].get_indices_shorter_than(None))
    loaders = [torch.utils.data.DataLoader(d, batch_size=batch_size, sampler=train_sampler if not i else None, pin_memory=True, collate_fn=collate_fn) for i, d in enumerate(datasets)]
    return (encoder, *loaders)


def set_time_limit(loader, time_limit):
    db = loader.dataset
    sampler = loader.sampler
    sampler.indices = db.get_indices_shorter_than(time_limit)


if __name__ == '__main__':
    import pprint
    train_load, val_load, test_load = get_dataloaders('TIMIT', 3)
    for (audio, lengths), sentence in train_load:
        print(audio.shape, audio)
        print()
        print(lengths.shape, lengths)
        print()
        print(sentence)
        break
