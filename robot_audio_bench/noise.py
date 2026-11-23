"""バブル雑音（LibriSpeech）とピンクノイズの生成。

numpy / soundfile は関数内で遅延 import する。soundfile が無いときは
WAV だけを標準ライブラリで読み、FLAC はエラー文で案内する。
"""
from __future__ import annotations

import array
import math
import sys
import wave
from pathlib import Path

from .wav import SAMPLE_RATE, write_wav

AUDIO_SUFFIXES = (".flac", ".wav")
DEFAULT_TARGET_RMS = 0.1  # -20 dBFS
FLAC_HINT = (
    "FLAC を読むには soundfile が必要（pip install soundfile）。"
    "または事前に 16 kHz mono WAV へ変換して --librispeech に置く。")


def speaker_id_from_path(path):
    """LibriSpeech の話者 ID を推定する。

    LibriSpeech は `<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac`。
    ファイル名に '-' が無い場合は親ディレクトリ名を話者とみなす。
    """
    stem = Path(path).stem
    if "-" in stem:
        candidate = stem.split("-", 1)[0]
        if candidate:
            return candidate
    return Path(path).parent.name


def collect_speakers(root, suffixes=AUDIO_SUFFIXES):
    """ディレクトリ以下を走査し、話者 ID -> 音声ファイルのリストを返す。"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"LibriSpeech のディレクトリが無い: {root}")
    groups = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            groups.setdefault(speaker_id_from_path(path), []).append(path)
    return {speaker: files for speaker, files in sorted(groups.items()) if files}


def _resample_linear(samples, source_rate, target_rate=SAMPLE_RATE):
    import numpy as np
    if source_rate == target_rate:
        return samples
    count = int(round(len(samples) * target_rate / float(source_rate)))
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    source_index = np.arange(len(samples), dtype=np.float64)
    target_index = np.arange(count, dtype=np.float64) * (source_rate / float(target_rate))
    return np.interp(target_index, source_index, samples).astype(np.float32)


def _read_wav_stdlib(path):
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        rate = stream.getframerate()
        frames = stream.getnframes()
        payload = stream.readframes(frames)
    if width != 2:
        raise ValueError(f"WAV は int16 だけ扱える: {path} (sampwidth={width * 8} bit)")
    values = array.array("h")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    if channels <= 1:
        return [value / 32768.0 for value in values], rate
    mono = [
        sum(values[index + channel] for channel in range(channels)) / (channels * 32768.0)
        for index in range(0, len(values), channels)
    ]
    return mono, rate


def read_audio(path):
    """音声ファイルを (mono float サンプル, rate) で読む。soundfile 優先。"""
    path = Path(path)
    try:
        import soundfile  # 遅延 import
    except ImportError:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(FLAC_HINT) from None
        return _read_wav_stdlib(path)
    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    return data, int(rate)


def _speaker_signal(files, sample_count):
    """1 話者の発話を連結し、sample_count に揃える（足りなければ繰り返す）。"""
    import numpy as np
    chunks = []
    total = 0
    while total < sample_count:
        for path in files:
            samples, rate = read_audio(path)
            samples = _resample_linear(np.asarray(samples, dtype=np.float32), rate)
            if len(samples) == 0:
                continue
            chunks.append(samples)
            total += len(samples)
            if total >= sample_count:
                break
        else:
            if not chunks:
                raise ValueError(f"話者の音声が空: {files[0]}")
            continue
    return np.concatenate(chunks)[:sample_count].astype(np.float32)


def make_babble(librispeech_dir, speakers=8, duration=600.0, seed=0,
                rate=SAMPLE_RATE, target_rms=DEFAULT_TARGET_RMS, output=None):
    """LibriSpeech から話者の異なる発話を混ぜてバブル雑音を作る。

    8 人以上（--speakers 人）の話者を選び、各話者の発話を連結して同じ長さに
    揃え、足し合わせて RMS 正規化した 16 kHz mono int16 WAV を出力する。
    """
    import numpy as np  # 遅延 import（numpy 必須）

    groups = collect_speakers(librispeech_dir)
    speaker_count = int(speakers)
    if speaker_count <= 0:
        raise ValueError("--speakers は 1 以上")
    if len(groups) < speaker_count:
        raise ValueError(
            f"話者が足りない: {len(groups)} 人しか見つからない（{speaker_count} 人必要）")
    selected = sorted(groups)[:]
    import random
    random.Random(seed).shuffle(selected)
    selected = selected[:speaker_count]

    sample_count = int(round(float(duration) * rate))
    if sample_count <= 0:
        raise ValueError("--duration は正の値")
    mix = np.zeros(sample_count, dtype=np.float64)
    for speaker in selected:
        mix += _speaker_signal(groups[speaker], sample_count)

    mix -= mix.mean()
    rms = float(np.sqrt(np.mean(mix * mix)))
    if not math.isfinite(rms) or rms <= 0.0:
        raise ValueError("バブル雑音の RMS が 0 になった（入力が無音）")
    scale = float(target_rms) / rms
    peak = float(np.max(np.abs(mix))) * scale
    if peak > 0.99:
        scale *= 0.99 / peak
    mix *= scale
    samples = np.clip(np.round(mix * 32767.0), -32768, 32767).astype(np.int16)
    if output is not None:
        write_wav(output, samples.tolist(), rate=rate)
    return {
        "speakers": selected,
        "speaker_count": len(selected),
        "duration_sec": sample_count / float(rate),
        "rate_hz": rate,
        "rms": target_rms,
    }


def generate_pink_noise(duration=10.0, seed=0, rate=SAMPLE_RATE,
                        target_rms=DEFAULT_TARGET_RMS):
    """ピンクノイズの int16 サンプル列を作る（numpy 必須）。"""
    import numpy as np

    sample_count = int(round(float(duration) * rate))
    if sample_count <= 0:
        raise ValueError("--duration は正の値")
    random = np.random.default_rng(seed)
    white = random.standard_normal(sample_count)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / rate)
    shaping = np.ones_like(frequencies)
    shaping[1:] = 1.0 / np.sqrt(frequencies[1:])
    shaping[0] = 0.0
    pink = np.fft.irfft(spectrum * shaping, n=sample_count)
    pink -= pink.mean()
    rms = float(np.sqrt(np.mean(pink * pink)))
    if not math.isfinite(rms) or rms <= 0.0:
        raise ValueError("ピンクノイズの RMS が 0 になった")
    pink *= float(target_rms) / rms
    peak = float(np.max(np.abs(pink)))
    if peak > 0.99:
        pink *= 0.99 / peak
    return np.clip(np.round(pink * 32767.0), -32768, 32767).astype(np.int16)


def make_pink(duration=10.0, seed=0, rate=SAMPLE_RATE, output=None):
    samples = generate_pink_noise(duration=duration, seed=seed, rate=rate)
    if output is not None:
        write_wav(output, samples.tolist(), rate=rate)
    return {
        "duration_sec": len(samples) / float(rate),
        "rate_hz": rate,
        "seed": seed,
    }


# 短い別名。
babble = make_babble
pink_noise = generate_pink_noise
