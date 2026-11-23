"""ピンクノイズ録音の 1/3 オクターブ指向性と supercardioid の理論値。"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from .signal import segment
from .wav import SAMPLE_RATE, read_wav

# IEC 61260 の公称 1/3 オクターブ中心周波数（125 Hz〜6.3 kHz）。
THIRD_OCTAVE_CENTERS = [
    125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
    1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
]
DIRECTIVITY_COLUMNS = ["theta_deg", "band_hz", "measured_db", "theory_db", "diff_db"]


def third_octave_centers():
    return list(THIRD_OCTAVE_CENTERS)


def third_octave_edges(center_hz):
    factor = 2.0 ** (1.0 / 6.0)
    return center_hz / factor, center_hz * factor


def supercardioid_db(theta_deg):
    """supercardioid の理論値 20*log10(|0.37 + 0.63*cos(theta)|)。

    ヌル（値が 0）のときは -inf ではなく None（CSV では空欄）。
    """
    value = abs(0.37 + 0.63 * math.cos(math.radians(float(theta_deg))))
    if value <= 1e-12:
        return None
    return 20.0 * math.log10(value)


def band_powers(samples, rate=SAMPLE_RATE):
    """1/3 オクターブ帯と全帯域（125 Hz〜6.3 kHz）のパワーを返す。"""
    import numpy as np  # 遅延 import

    values = np.asarray(samples, dtype=np.float64)
    if len(values) < 2:
        raise ValueError("指向性の計算には 2 サンプル以上必要")
    window = np.hanning(len(values))
    spectrum = np.fft.rfft(values * window)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / rate)
    normalization = float(np.sum(window * window) * len(values))
    powers = {}
    broadband = 0.0
    for center in THIRD_OCTAVE_CENTERS:
        low, high = third_octave_edges(center)
        mask = (frequencies >= low) & (frequencies < high)
        value = float(np.sum(power[mask])) / normalization
        powers[center] = value
        broadband += value
    powers["broadband"] = broadband
    return powers


def compute_directivity(recordings, start_s=0.0, duration_s=10.0,
                        reference_deg=0.0, rate=SAMPLE_RATE):
    """角度 -> サンプル列から指向性の行を作る。

    recordings は {theta_deg: samples}。0°（reference_deg）を基準にした
    dB 差を帯域ごとと全帯域で出す。
    """
    if not recordings:
        raise ValueError("録音が 1 つも無い")
    if float(reference_deg) not in recordings:
        raise ValueError(f"基準角 {reference_deg}° の録音が無い: {sorted(recordings)}")
    powers = {}
    for theta, samples in recordings.items():
        cropped = segment(samples, start_s, start_s + duration_s, rate=rate)
        powers[theta] = band_powers(cropped, rate=rate)
    reference = powers[float(reference_deg)]
    reference_theory = supercardioid_db(reference_deg)
    rows = []
    for theta in sorted(recordings):
        measured = powers[theta]
        theta_theory = supercardioid_db(theta)
        theory_db = (theta_theory - reference_theory
                     if theta_theory is not None and reference_theory is not None else None)
        for band in ["broadband"] + THIRD_OCTAVE_CENTERS:
            reference_power = reference[band]
            power = measured[band]
            measured_db = (10.0 * math.log10(power / reference_power)
                           if power > 0.0 and reference_power > 0.0 else None)
            difference = (measured_db - theory_db
                          if measured_db is not None and theory_db is not None else None)
            rows.append({
                "theta_deg": theta,
                "band_hz": band,
                "measured_db": measured_db,
                "theory_db": theory_db,
                "diff_db": difference,
            })
    return rows


def write_directivity_csv(rows, path):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DIRECTIVITY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "theta_deg": row["theta_deg"],
                "band_hz": row["band_hz"],
                "measured_db": "" if row["measured_db"] is None else f"{row['measured_db']:.4f}",
                "theory_db": "" if row["theory_db"] is None else f"{row['theory_db']:.4f}",
                "diff_db": "" if row["diff_db"] is None else f"{row['diff_db']:.4f}",
            })


def parse_recording_spec(spec):
    """'90=rec90.wav' を (90.0, Path) にする。"""
    if "=" not in spec:
        raise ValueError(f"--rec は THETA=PATH の形式: {spec!r}")
    theta_text, path_text = spec.split("=", 1)
    try:
        theta = float(theta_text)
    except ValueError as error:
        raise ValueError(f"--rec の角度が数値ではない: {spec!r}") from error
    if not path_text:
        raise ValueError(f"--rec のパスが空: {spec!r}")
    return theta, Path(path_text)


def load_recordings(specs, rate=SAMPLE_RATE):
    recordings = {}
    for spec in specs:
        theta, path = parse_recording_spec(spec)
        if theta in recordings:
            raise ValueError(f"角度 {theta}° が重複している")
        recordings[theta] = read_wav(path, rate=rate)
    return recordings
