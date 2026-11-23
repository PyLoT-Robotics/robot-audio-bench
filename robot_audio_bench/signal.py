"""同期トーン検出・区間切り出し・SNR（標準ライブラリだけ）。"""
from __future__ import annotations

import math

from .wav import SAMPLE_RATE, seconds_to_samples

TONE_FREQUENCY_HZ = 1000.0
TONE_WINDOW_SEC = 0.02
TONE_SEARCH_SEC = 30.0
TONE_BASELINE_PERCENTILE = 0.10
TONE_THRESHOLD_FACTOR = 10.0  # 探索範囲の 10 パーセンタイルの +10 dB
TONE_MIN_CONCENTRATION = 0.05  # 窓内パワーに占める 1 kHz 成分の下限（白色雑音の誤検出防止）
INT16_SCALE = 32768.0


def goertzel_power(samples, rate=SAMPLE_RATE, frequency_hz=TONE_FREQUENCY_HZ):
    """Goertzel 法で 1 周波数のパワー（平均二乗に比例する値）を出す。"""
    count = len(samples)
    if count == 0 or rate <= 0:
        return 0.0
    omega = 2.0 * math.pi * frequency_hz / rate
    coefficient = 2.0 * math.cos(omega)
    previous = 0.0
    previous_two = 0.0
    for value in samples:
        current = float(value) + coefficient * previous - previous_two
        previous_two, previous = previous, current
    power = (previous * previous + previous_two * previous_two
             - coefficient * previous * previous_two)
    return max(0.0, power) / (float(count) * float(count))


def percentile(values, fraction):
    """線形補間の分位点（numpy を使わない）。空なら None。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(float(fraction), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def detect_tone_start(samples, rate=SAMPLE_RATE, frequency_hz=TONE_FREQUENCY_HZ,
                      search_sec=TONE_SEARCH_SEC, window_sec=TONE_WINDOW_SEC,
                      baseline_percentile=TONE_BASELINE_PERCENTILE,
                      threshold_factor=TONE_THRESHOLD_FACTOR):
    """録音の先頭 search_sec から 1 kHz トーンの開始時刻 [s] を探す。

    20 ms 窓ごとの Goertzel パワーを出し、探索範囲（先頭 30 s）の全窓の
    10 パーセンタイルを基準にする。その 10 倍（+10 dB）を初めて超えた窓の
    開始時刻を返す。見つからなければ None。VAD は使わない。

    録音開始直後にトーンが鳴っても、基準が全窓の下位 10 % なので
    トーン窓に引きずられない。
    """
    window_samples = seconds_to_samples(window_sec, rate)
    if window_samples <= 0:
        raise ValueError("window_sec は正の値にする")
    search_samples = min(len(samples), seconds_to_samples(search_sec, rate))
    window_count = search_samples // window_samples
    if window_count == 0:
        return None
    powers = []
    mean_squares = []
    for index in range(window_count):
        window = samples[index * window_samples:(index + 1) * window_samples]
        powers.append(goertzel_power(window, rate, frequency_hz))
        mean_squares.append(sum(float(value) * float(value) for value in window) / len(window))
    baseline = percentile(powers, baseline_percentile)
    threshold = baseline * threshold_factor
    if baseline <= 0.0:
        # 探索範囲がデジタル無音（または無音に近い）なら、最初にエネルギーが出た窓。
        threshold = 0.0
    for index, power in enumerate(powers):
        if power <= threshold or mean_squares[index] <= 0.0:
            continue
        # 白色雑音は 20 ms 窓の 1 kHz 成分が窓全体の 1 % 程度なので、
        # パワーが基準を超えてもトーンらしさ（1 kHz への集中）を要求する。
        if power / mean_squares[index] >= TONE_MIN_CONCENTRATION:
            return index * window_sec
    return None


def segment(samples, start_sec, end_sec, rate=SAMPLE_RATE):
    """[start_sec, end_sec) を切り出す。録音の外にはみ出したら ValueError。"""
    start = seconds_to_samples(start_sec, rate)
    end = seconds_to_samples(end_sec, rate)
    if start < 0 or end < start:
        raise ValueError(f"区間が不正: [{start_sec}, {end_sec})")
    if end > len(samples):
        raise ValueError(
            f"録音が短い: [{start_sec:.3f}, {end_sec:.3f}) s が必要だが "
            f"{len(samples) / float(rate):.3f} s しかない")
    return samples[start:end]


def mean_square(samples):
    if not samples:
        return 0.0
    return sum(float(value) * float(value) for value in samples) / len(samples)


def power_to_dbfs(power):
    """平均二乗（int16 スケール）を dBFS にする。0 なら None。"""
    if power <= 0.0:
        return None
    normalized = power / (INT16_SCALE * INT16_SCALE)
    return 10.0 * math.log10(normalized)


def snr_from_segments(noise_samples, speech_samples):
    """(SNR dB, 有効か, noise_dbfs, speech_dbfs) を返す。

    P_n = 雑音区間の平均二乗、P_sn = 発話区間の平均二乗、
    SNR = 10*log10((P_sn - P_n) / P_n)。
    P_n == 0 または P_sn <= P_n なら SNR は None（空欄）で invalid。
    """
    noise_power = mean_square(noise_samples)
    speech_power = mean_square(speech_samples)
    noise_dbfs = power_to_dbfs(noise_power)
    speech_dbfs = power_to_dbfs(speech_power)
    if noise_power <= 0.0 or speech_power <= noise_power:
        return None, False, noise_dbfs, speech_dbfs
    return 10.0 * math.log10((speech_power - noise_power) / noise_power), True, noise_dbfs, speech_dbfs


# 短い別名（テスト・対話用）。
compute_snr = snr_from_segments
detect_offset = detect_tone_start
