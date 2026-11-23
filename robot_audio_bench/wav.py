"""16 kHz・モノラル・int16 WAV の読み書き（標準ライブラリだけ）。

ベンチの録音・刺激・再生リストはすべて 16 kHz mono int16 を前提にする。
リサンプルはしない。違う形式ははっきりした ValueError にする。
"""
from __future__ import annotations

import array
import math
import sys
import wave
from pathlib import Path

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
INT16_MIN = -32768
INT16_MAX = 32767


class WavFormatError(ValueError):
    """16 kHz mono int16 以外の WAV を渡されたときのエラー。"""


def wav_info(path):
    """WAV の (frames, rate, channels, sampwidth) を読む（PCM は読まない）。"""
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes(), wav.getframerate(), wav.getnchannels(), wav.getsampwidth()


def check_wav_format(path, rate=None, channels=None, sampwidth=None):
    """WAV の形式を確かめる。違えば WavFormatError。"""
    frames, actual_rate, actual_channels, actual_width = wav_info(path)
    expected_rate = SAMPLE_RATE if rate is None else rate
    expected_channels = CHANNELS if channels is None else channels
    expected_width = SAMPLE_WIDTH if sampwidth is None else sampwidth
    if (actual_rate, actual_channels, actual_width) != (
            expected_rate, expected_channels, expected_width):
        raise WavFormatError(
            f"16 kHz モノラル int16 の WAV が必要: {path} "
            f"(rate={actual_rate} Hz, channels={actual_channels}, "
            f"sampwidth={actual_width * 8} bit; "
            f"期待値={expected_rate} Hz, {expected_channels} ch, {expected_width * 8} bit)")
    return frames, actual_rate


def read_wav(path, rate=None):
    """16 kHz mono int16 の WAV を int16 サンプルのリストとして読む。"""
    expected_rate = SAMPLE_RATE if rate is None else rate
    frames, actual_rate = check_wav_format(path, rate=expected_rate)
    with wave.open(str(path), "rb") as wav:
        payload = wav.readframes(frames)
    samples = array.array("h")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    return list(samples)


def write_wav(path, samples, rate=SAMPLE_RATE):
    """int16 サンプル列を 16 kHz mono WAV に書く（範囲外はクリップ）。"""
    path = Path(path)
    values = array.array("h", (max(INT16_MIN, min(INT16_MAX, int(round(value))))
                               for value in samples))
    if sys.byteorder != "little":
        values.byteswap()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(rate)
        wav.writeframes(values.tobytes())
    return len(values)


def duration_sec(path):
    """WAV の長さ [s]。"""
    frames, rate, _, _ = wav_info(path)
    return frames / float(rate)


def seconds_to_samples(seconds, rate=SAMPLE_RATE):
    """秒をサンプル数に丸める（再生リストと切り出しで同じ規則を使う）。"""
    return int(round(float(seconds) * rate))


def samples_to_seconds(samples, rate=SAMPLE_RATE):
    return samples / float(rate)


def silence(frames):
    return array.array("h", [0]) * max(0, int(frames))


def tone(frequency_hz, seconds, rate=SAMPLE_RATE, amplitude=0.5):
    """テスト用の正弦波（1 kHz 同期トーンなど）。"""
    count = seconds_to_samples(seconds, rate)
    scale = max(0.0, min(1.0, float(amplitude))) * INT16_MAX
    return [int(round(scale * math.sin(2.0 * math.pi * frequency_hz * index / rate)))
            for index in range(count)]
