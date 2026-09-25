"""Frozen offline LID teachers behind one interface.

Every teacher returns, per clip, a probability vector over ROUTE = LANGS + ["other"]:
its native posterior with Urdu folded into Hindi and every language outside LANGS
summed into "other". `restrict()` renormalises over LANGS only (the routing set), which
is the same as the "allowed_langs" hard filter the Indic-Transcribe card describes.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from slid.audio import SR
from slid.config import LANGS, MERGE_INTO

ROUTE = LANGS + ["other"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fold(labels: list[str], probs: np.ndarray) -> np.ndarray:
    """[B, n_native] native posteriors -> [B, len(ROUTE)]."""
    out = np.zeros((probs.shape[0], len(ROUTE)), dtype=np.float64)
    for j, lab in enumerate(labels):
        lab = MERGE_INTO.get(lab, lab)
        out[:, ROUTE.index(lab) if lab in LANGS else -1] += probs[:, j]
    return out / out.sum(axis=1, keepdims=True)


def restrict(p: np.ndarray) -> np.ndarray:
    q = p[..., : len(LANGS)]
    return q / q.sum(axis=-1, keepdims=True)


def _pad(wavs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    lens = torch.tensor([len(w) for w in wavs])
    x = torch.zeros(len(wavs), int(lens.max()))
    for i, w in enumerate(wavs):
        x[i, : len(w)] = torch.from_numpy(w)
    return x, lens


class Teacher:
    name = "base"
    batch_size = 16

    def native(self, wavs: list[np.ndarray]) -> tuple[list[str], np.ndarray]:
        raise NotImplementedError

    @torch.inference_mode()
    def probs(self, wavs: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(wavs), self.batch_size):
            labels, p = self.native(wavs[i: i + self.batch_size])
            out.append(fold(labels, p))
        return np.concatenate(out)


class Ecapa(Teacher):
    name = "ecapa"

    def __init__(self):
        from speechbrain.inference.classifiers import EncoderClassifier
        self.m = EncoderClassifier.from_hparams("speechbrain/lang-id-voxlingua107-ecapa",
                                                savedir="pretrained_models/ecapa",
                                                run_opts={"device": DEVICE})
        enc = self.m.hparams.label_encoder
        self.labels = [enc.ind2lab[i].split(":")[0].strip() for i in range(len(enc.ind2lab))]

    def native(self, wavs):
        x, lens = _pad(wavs)
        logp = self.m.classify_batch(x.to(DEVICE), (lens / lens.max()).to(DEVICE))[0]
        return self.labels, logp.exp().float().cpu().numpy()


class AmberNet(Teacher):
    name = "ambernet"

    def __init__(self):
        path = snapshot_download("surogate/ambernet-langid")
        sys.path.insert(0, path)
        from modeling_ambernet import AmberNet as _AmberNet
        self.m = _AmberNet.from_pretrained(path).to(DEVICE)
        self.labels = ["he" if l == "iw" else "jv" if l == "jw" else l for l in self.m.labels]

    def native(self, wavs):
        x, lens = _pad(wavs)
        logits = self.m(x.to(DEVICE), lens.to(DEVICE))[0]
        return self.labels, logits.float().softmax(-1).cpu().numpy()


class Whisper(Teacher):
    """Language posterior = softmax over language tokens at the first decoder step."""
    name = "whisper-turbo"
    batch_size = 8

    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
        from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        self.fe = WhisperFeatureExtractor.from_pretrained(model_id)
        self.m = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(DEVICE).eval()
        lang_to_id = self.m.generation_config.lang_to_id
        self.labels = [t.strip("<|>") for t in lang_to_id]
        self.lang_ids = torch.tensor(list(lang_to_id.values()), device=DEVICE)
        self.sot = self.m.generation_config.decoder_start_token_id

    def native(self, wavs):
        feats = self.fe(wavs, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(DEVICE, self.m.dtype)
        dec = torch.full((len(wavs), 1), self.sot, device=DEVICE)
        logits = self.m(input_features=feats, decoder_input_ids=dec).logits[:, -1]
        return self.labels, logits[:, self.lang_ids].float().softmax(-1).cpu().numpy()


class IndicTranscribe(Teacher):
    """Bodhan/AI4Bharat Indic-Transcribe-core: one decoder step after the encoder."""
    name = "indic-transcribe"
    batch_size = 1

    def __init__(self):
        path = snapshot_download("bodhan-ai/indic-transcribe-core")
        sys.path.insert(0, path)
        from indic_transcribe import IndicTranscribe as _IT
        self.asr = _IT.from_pretrained(path, device=DEVICE)

    def native(self, wavs):
        top = self.asr.identify(wavs[0], topk=10_000)
        labels, p = zip(*top)
        return list(labels), np.array([p], dtype=np.float64)


class XlsrVoxLingua(Teacher):
    """TalTechNLP XLS-R-300M + attentive pooling, rebuilt without their old SpeechBrain yaml."""
    name = "xlsr-voxlingua"
    batch_size = 8

    def __init__(self):
        from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
        from speechbrain.lobes.models.Xvector import Classifier
        from transformers import AutoConfig, Wav2Vec2Model
        path = Path(snapshot_download("TalTechNLP/voxlingua107-xls-r-300m-wav2vec"))
        self.w2v = Wav2Vec2Model(AutoConfig.from_pretrained("facebook/wav2vec2-xls-r-300m"))
        state = torch.load(path / "wav2vec2.ckpt", map_location="cpu")
        state = {k.removeprefix("model."): v for k, v in state.items()}
        missing, unexpected = self.w2v.load_state_dict(state, strict=False)
        assert not [k for k in missing if "masked_spec_embed" not in k], missing
        self.pool = AttentiveStatisticsPooling(1024, attention_channels=64)
        self.clf = Classifier(input_shape=[None, None, 2048], activation=torch.nn.LeakyReLU,
                              lin_blocks=1, lin_neurons=512, out_neurons=107)
        mods = torch.nn.ModuleList([self.pool, self.clf])
        mods.load_state_dict(torch.load(path / "model.ckpt", map_location="cpu"))
        for m in (self.w2v, self.pool, self.clf):
            m.to(DEVICE).eval()
        lab2ind = {}
        for line in (path / "label_encoder.txt").read_text().splitlines():
            if line.startswith("'") and "=>" in line and not line.startswith("'starting_index'"):
                lab, ind = line.split("=>")
                lab2ind[lab.strip().strip("'")] = int(ind)
        self.labels = [lab for lab, _ in sorted(lab2ind.items(), key=lambda kv: kv[1])][:107]

    def native(self, wavs):
        x, lens = _pad(wavs)
        h = self.w2v(x.to(DEVICE)).last_hidden_state
        h = F.layer_norm(h, h.shape[-1:])
        pooled = self.pool(h.transpose(1, 2), (lens / lens.max()).to(DEVICE)).transpose(1, 2)
        logits = self.clf(pooled).squeeze(1)
        return self.labels, logits.float().softmax(-1).cpu().numpy()


TEACHERS = {t.name: t for t in (Ecapa, AmberNet, Whisper, IndicTranscribe, XlsrVoxLingua)}


def load_teacher(name: str) -> Teacher:
    return TEACHERS[name]()
