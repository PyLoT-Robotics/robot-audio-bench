"""試行順のシャッフルと再生用 playlist WAV / CSV の生成。"""
from __future__ import annotations

import array
import csv
import random
import wave
from pathlib import Path

from .wav import SAMPLE_RATE, check_wav_format, read_wav, silence, tone

TONE_SEC = 0.5
LEAD_SILENCE_SEC = 1.0
NOISE_SEC = 2.0
GAP_SEC = 1.0
TONE_FREQUENCY_HZ = 1000.0

PLAYLIST_COLUMNS = [
    "trial_idx", "utt_id", "speaker", "lang", "keyword", "text",
    "noise_start_s", "noise_end_s", "speech_start_s", "speech_end_s",
]


def _seconds(samples):
    return samples / float(SAMPLE_RATE)


def _write_frames(stream, samples):
    values = array.array("h", [int(value) for value in samples])
    stream.writeframes(values.tobytes())


def parse_speakers(value):
    speakers = [item.strip() for item in str(value).split(",") if item.strip()]
    if not speakers:
        raise ValueError("--speakers が空")
    return speakers


def plan_playlist(trials):
    """シャッフル済み試行に、トーン開始を 0 とするサンプル位置を割り当てる。

    先頭は 1 kHz・0.5 s のトーン + 1.0 s の無音。各試行は雑音区間 2.0 s
    （WAV 上は無音。雑音は別再生）→発話。試行の間には 1.0 s の間隔を置く。
    最後の試行の後には間隔を置かない（WAV の長さ = 最後の speech_end）。
    """
    tone_samples = int(round(TONE_SEC * SAMPLE_RATE))
    lead_samples = int(round(LEAD_SILENCE_SEC * SAMPLE_RATE))
    noise_samples = int(round(NOISE_SEC * SAMPLE_RATE))
    gap_samples = int(round(GAP_SEC * SAMPLE_RATE))

    cursor = tone_samples + lead_samples
    positions = []
    for index, trial in enumerate(trials):
        noise_start = cursor
        noise_end = noise_start + noise_samples
        speech_start = noise_end
        speech_end = speech_start + trial["frames"]
        positions.append({
            "noise_start": noise_start, "noise_end": noise_end,
            "speech_start": speech_start, "speech_end": speech_end,
        })
        cursor = speech_end
        if index + 1 < len(trials):
            cursor += gap_samples
    return positions, cursor


def build_playlist(stimuli_rows, wav_dir, speakers, lang, seed=0,
                   output_wav=None, output_csv=None):
    """stimuli.csv の行と録音済み WAV から playlist を作る。

    WAV は 16 kHz mono int16 を前提にし、違えば WavFormatError。
    戻り値は CSV と同じ列を持つ dict のリスト。
    """
    wav_dir = Path(wav_dir)
    selected = [row for row in stimuli_rows if row.get("lang") == lang]
    if not selected:
        raise ValueError(f"stimuli.csv に lang={lang!r} の行が無い")
    if not speakers:
        raise ValueError("speaker が 1 人も指定されていない")

    trials = []
    missing = []
    for row in selected:
        for speaker in speakers:
            path = wav_dir / f"{row['utt_id']}_{speaker}.wav"
            if not path.is_file():
                missing.append(str(path))
                continue
            frames, _ = check_wav_format(path)
            trials.append({
                "utt_id": row["utt_id"], "speaker": speaker, "lang": row["lang"],
                "keyword": row["keyword"], "text": row["text"],
                "path": path, "frames": frames,
            })
    if missing:
        preview = "\n  ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n  ... 他 {len(missing) - 20} 件"
        raise FileNotFoundError(f"録音 WAV が無い:\n  {preview}{suffix}")

    random.Random(seed).shuffle(trials)
    positions, total_frames = plan_playlist(trials)

    rows = []
    for index, (trial, position) in enumerate(zip(trials, positions)):
        rows.append({
            "trial_idx": index,
            "utt_id": trial["utt_id"],
            "speaker": trial["speaker"],
            "lang": trial["lang"],
            "keyword": trial["keyword"],
            "text": trial["text"],
            "noise_start_s": _seconds(position["noise_start"]),
            "noise_end_s": _seconds(position["noise_end"]),
            "speech_start_s": _seconds(position["speech_start"]),
            "speech_end_s": _seconds(position["speech_end"]),
        })

    if output_wav is not None:
        output_wav = Path(output_wav)
        with wave.open(str(output_wav), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(SAMPLE_RATE)
            _write_frames(stream, tone(TONE_FREQUENCY_HZ, TONE_SEC))
            _write_frames(stream, silence(int(round(LEAD_SILENCE_SEC * SAMPLE_RATE))))
            for index, trial in enumerate(trials):
                _write_frames(stream, silence(int(round(NOISE_SEC * SAMPLE_RATE))))
                _write_frames(stream, read_wav(trial["path"]))
                if index + 1 < len(trials):
                    _write_frames(stream, silence(int(round(GAP_SEC * SAMPLE_RATE))))
            written = stream.tell()
        if written != total_frames:
            raise RuntimeError(f"playlist の長さが計画と違う: {written} != {total_frames}")

    if output_csv is not None:
        write_playlist_csv(rows, output_csv)
    return rows


def write_playlist_csv(rows, path):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PLAYLIST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("noise_start_s", "noise_end_s", "speech_start_s", "speech_end_s"):
                formatted[key] = f"{float(row[key]):.6f}"
            writer.writerow(formatted)


def read_playlist_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
