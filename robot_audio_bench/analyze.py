"""録音から試行を切り出し、SNR と whisper 認識結果を trials.csv にまとめる。"""
from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path

from .config import english_prompts, load_hri_config
from .playlist import read_playlist_csv
from .signal import detect_tone_start, segment, snr_from_segments
from .text_metrics import (
    character_error_rate, infer_in_list, infer_kind, is_correct, word_error_rate,
)
from .wav import read_wav, write_wav
from .whisper import (
    cache_key, load_whisper_config, parse_whisper_json,
    whisper_args_from_config, whisper_args_from_yaml,
)

TRIAL_COLUMNS = [
    "trial_idx", "utt_id", "speaker", "lang", "kind", "keyword", "in_list",
    "theta_deg", "yaw_deg", "distance_m", "noise", "snr_db", "snr_valid",
    "noise_dbfs", "speech_dbfs", "hyp", "ref", "prompt", "wer", "cer",
    "correct", "mean_logprob", "n_tokens", "offset_s", "recording",
]

# 再生側と録音側の時計のずれで発話の頭が雑音区間に入るのを避けるため、
# SNR の雑音区間は終わりを 0.1 s 手前で切る。
NOISE_GUARD_SEC = 0.1


def _float(value, digits=4):
    return "" if value is None else f"{float(value):.{digits}f}"


def _bool(value):
    return "true" if bool(value) else "false"


def _run_whisper(whisper_cli, arguments, speech_path, prefix):
    command = [str(whisper_cli)] + list(arguments) + [
        "-ojf", "-np", "-f", str(speech_path), "-of", str(prefix)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"whisper-cli が見つからない: {whisper_cli} "
            "(--whisper-cli でパスを指定する。README の手順で whisper.cpp をビルドする)") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"whisper-cli が失敗した (rc={completed.returncode}): "
            f"{completed.stderr.strip()[-2000:]}")
    json_path = Path(str(prefix) + ".json")
    if not json_path.is_file():
        raise RuntimeError(f"whisper-cli の JSON 出力が無い: {json_path}")
    return json_path.read_text(encoding="utf-8")


def _recognize(whisper_cli, arguments, speech_path, cache_directory):
    speech_bytes = Path(speech_path).read_bytes()
    if cache_directory is None:
        with tempfile.TemporaryDirectory() as temporary:
            text = _run_whisper(whisper_cli, arguments, speech_path,
                                Path(temporary) / "whisper")
            return parse_whisper_json(json.loads(text))
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / (cache_key(speech_bytes, arguments) + ".json")
    if cache_path.is_file():
        return parse_whisper_json(json.loads(cache_path.read_text(encoding="utf-8")))
    with tempfile.TemporaryDirectory() as temporary:
        text = _run_whisper(whisper_cli, arguments, speech_path,
                            Path(temporary) / "whisper")
    cache_path.write_text(text, encoding="utf-8")
    return parse_whisper_json(json.loads(text))


def _prompts_for_language(lang, hri_params, ja_prompt):
    """ASR 用プロンプトを先に解決する（設定不備はここでエラー）。

    日本語でも analyze は hri_params.yaml の存在を要求する（--no-asr のときだけ不要）。
    英語は hri_params.yaml の name_prompt / drink_prompt（空なら候補列挙文）。
    """
    if str(lang).lower() == "ja":
        load_hri_config(hri_params, strict=True)
        return {"name": ja_prompt or "", "drink": ja_prompt or ""}
    name_prompt, drink_prompt = english_prompts(hri_params, strict=True)
    return {"name": name_prompt, "drink": drink_prompt}


def analyze_recording(recording, playlist_rows, theta_deg, distance_m, noise, lang,
                      yaw_deg=None, offset_s=None, whisper_cli="whisper-cli",
                      model=None, whisper_yaml=None, hri_params=None, ja_prompt="",
                      asr_cache_dir=None, no_asr=False):
    """1 本の録音と playlist CSV の行から trials.csv の行を作る。

    戻り値は (rows, offset_s)。offset_s が None なら録音先頭 30 s から
    1 kHz トーンを探す。見つからなければ ValueError。
    --no-asr 以外では whisper.yaml / hri_params.yaml が読めなければ
    ConfigurationError（設定パスは既定ならリポジトリ内を探す）。
    """
    prompts = None
    arguments_by_kind = {}
    if not no_asr:
        whisper_config = load_whisper_config(whisper_yaml, strict=True)
        prompts = _prompts_for_language(lang, hri_params, ja_prompt)
        # yaml は 1 回だけ読み、kind ごとの引数もループの前に 1 回作る。
        arguments_by_kind = {
            kind: whisper_args_from_config(
                whisper_config, lang, prompts[kind], model=model)
            for kind in ("name", "drink")
        }

    samples = read_wav(recording)
    actual_offset = offset_s
    if actual_offset is None:
        actual_offset = detect_tone_start(samples)
        if actual_offset is None:
            raise ValueError(
                f"録音の先頭 30 s に 1 kHz の同期トーンが見つからない: {recording}")
    actual_offset = float(actual_offset)

    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for index, trial in enumerate(playlist_rows):
            noise_start = actual_offset + float(trial["noise_start_s"])
            noise_end = max(
                noise_start,
                actual_offset + float(trial["noise_end_s"]) - NOISE_GUARD_SEC)
            speech_start = actual_offset + float(trial["speech_start_s"])
            speech_end = actual_offset + float(trial["speech_end_s"])
            noise_samples = segment(samples, noise_start, noise_end)
            speech_samples = segment(samples, speech_start, speech_end)
            snr_db, snr_valid, noise_dbfs, speech_dbfs = snr_from_segments(
                noise_samples, speech_samples)

            utt_id = str(trial["utt_id"])
            kind = infer_kind(utt_id, trial.get("text", ""))
            in_list = infer_in_list(utt_id)
            hyp = ""
            prompt = ""
            wer = ""
            cer = ""
            correct = ""
            mean_logprob = ""
            n_tokens = ""
            if not no_asr:
                speech_path = temporary / f"trial_{index:04d}.wav"
                write_wav(speech_path, speech_samples)
                prompt = prompts[kind]
                arguments = arguments_by_kind[kind]
                hyp, logprob, token_count = _recognize(
                    whisper_cli, arguments, speech_path, asr_cache_dir)
                mean_logprob = _float(logprob, 6)
                n_tokens = str(token_count)
                correct = _bool(is_correct(trial["keyword"], hyp, lang))
                if str(lang).lower() == "ja":
                    cer = _float(character_error_rate(trial["text"], hyp), 4)
                else:
                    wer = _float(word_error_rate(trial["text"], hyp), 4)

            rows.append({
                "trial_idx": trial.get("trial_idx", index),
                "utt_id": utt_id,
                "speaker": trial.get("speaker", ""),
                "lang": trial.get("lang", lang),
                "kind": kind,
                "keyword": trial.get("keyword", ""),
                "in_list": _bool(in_list),
                "theta_deg": str(theta_deg),
                "yaw_deg": "" if yaw_deg is None else str(yaw_deg),
                "distance_m": str(distance_m),
                "noise": str(noise),
                "snr_db": _float(snr_db, 4),
                "snr_valid": _bool(snr_valid),
                "noise_dbfs": _float(noise_dbfs, 4),
                "speech_dbfs": _float(speech_dbfs, 4),
                "hyp": hyp,
                "ref": trial.get("text", ""),
                "prompt": prompt,
                "wer": wer,
                "cer": cer,
                "correct": correct,
                "mean_logprob": mean_logprob,
                "n_tokens": n_tokens,
                "offset_s": _float(actual_offset, 6),
                "recording": str(recording),
            })
    return rows, actual_offset


def write_trials_csv(rows, path, append=False):
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    if append and exists:
        with path.open("r", encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), None)
        if header != TRIAL_COLUMNS:
            raise ValueError(
                f"既存の trials.csv のヘッダが現在の列と違う: {path} "
                f"(既存={header}, 期待={TRIAL_COLUMNS})")
    mode = "a" if append and exists else "w"
    with path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRIAL_COLUMNS, lineterminator="\n")
        if not (append and exists):
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run_analysis(recording, playlist, output, theta_deg, distance_m, noise, lang,
                 yaw_deg=None, offset_s=None, whisper_cli="whisper-cli",
                 model=None, whisper_yaml=None, hri_params=None, ja_prompt="",
                 asr_cache_dir=None, no_asr=False, append=False):
    rows = read_playlist_csv(playlist)
    analyzed, actual_offset = analyze_recording(
        recording, rows, theta_deg, distance_m, noise, lang,
        yaw_deg=yaw_deg, offset_s=offset_s, whisper_cli=whisper_cli, model=model,
        whisper_yaml=whisper_yaml, hri_params=hri_params, ja_prompt=ja_prompt,
        asr_cache_dir=asr_cache_dir, no_asr=no_asr)
    write_trials_csv(analyzed, output, append=append)
    return analyzed, actual_offset
