"""whisper.cpp CLI の引数組み立てと JSON 出力の読み取り。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .config import (
    DEFAULT_HRI_PARAMS, ConfigurationError, english_prompts,
    resolve_repository_file,
)

DEFAULT_WHISPER_YAML = Path(__file__).resolve().parent / "config" / "whisper.yaml"

# whisper.yaml が読めないときの既定値（docker/config/whisper/whisper.yaml）。
DEFAULT_MODEL = "ggml-large-v3-turbo-q5_0.bin"
DEFAULT_BEAM_SIZE = 5
DEFAULT_BEST_OF = 5
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TEMPERATURE_INC = 0.2
DEFAULT_ENTROPY_THOLD = 2.4
DEFAULT_LOGPROB_THOLD = -1.0
DEFAULT_NO_SPEECH_THOLD = 0.6
DEFAULT_NO_CONTEXT = True
# whisper.cpp の n_max_text_ctx。CLI の -mc に渡す。0 にすると
# initial_prompt のトークンがデコーダに渡らないので 0 にしてはいけない。
DEFAULT_N_MAX_TEXT_CTX = 16384


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _whisper_candidate(path):
    if path is None:
        return resolve_repository_file(DEFAULT_WHISPER_YAML)
    candidate = Path(path)
    if not candidate.is_absolute() and candidate == DEFAULT_WHISPER_YAML:
        return resolve_repository_file(DEFAULT_WHISPER_YAML)
    return candidate


def load_whisper_config(path=None, strict=False):
    """whisper.yaml を読む。yaml は遅延 import。

    `path` が None（既定）なら resolve_repository_file() で探す。
    strict=True で見つからない・読めないときは ConfigurationError。
    それ以外は既定値を返し、loaded / error で失敗を伝える。
    """
    candidate = _whisper_candidate(path)
    parameters = {}
    loaded = False
    error_text = ""
    try:
        if candidate is None or not candidate.is_file():
            raise FileNotFoundError(
                f"whisper.yaml が見つからない: {candidate or DEFAULT_WHISPER_YAML}")
        import yaml
        with candidate.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        parameters = ((data.get("/whisper/whisper_node") or {}).get("ros__parameters") or {})
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(
                "whisper.yaml に /whisper/whisper_node.ros__parameters が無い")
        loaded = True
    except Exception as error:
        if strict:
            shown = candidate if candidate is not None else DEFAULT_WHISPER_YAML
            raise ConfigurationError(
                f"whisper.yaml を読めない: {shown} ({error})") from error
        parameters = {}
        error_text = str(error)
    model = _mapping(parameters.get("model"))
    sampling = _mapping(parameters.get("sampling"))
    transcription = _mapping(parameters.get("transcription"))
    decoding = _mapping(parameters.get("decoding"))
    model_path = str(model.get("path") or model.get("filename") or DEFAULT_MODEL)
    return {
        "model": model_path,
        "beam_size": sampling.get("beam_search_beam_size", DEFAULT_BEAM_SIZE),
        "best_of": sampling.get("greedy_best_of", DEFAULT_BEST_OF),
        "temperature": decoding.get("temperature", DEFAULT_TEMPERATURE),
        "temperature_inc": decoding.get("temperature_inc", DEFAULT_TEMPERATURE_INC),
        "entropy_thold": decoding.get("entropy_thold", DEFAULT_ENTROPY_THOLD),
        "logprob_thold": decoding.get("logprob_thold", DEFAULT_LOGPROB_THOLD),
        "no_speech_thold": decoding.get("no_speech_thold", DEFAULT_NO_SPEECH_THOLD),
        "n_max_text_ctx": transcription.get("n_max_text_ctx", DEFAULT_N_MAX_TEXT_CTX),
        "no_context": transcription.get("no_context", DEFAULT_NO_CONTEXT),
        "loaded": loaded,
        "source": str(candidate) if loaded else "",
        "error": error_text,
    }


def _format_number(value):
    """0.0 は "0"、-1.0 は "-1.0"、5 は "5" のように CLI 向けに整える。"""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if number == 0.0:
        return "0"
    if number.is_integer():
        return repr(number)
    return repr(number)


def whisper_args_from_config(config, lang, prompt, model=None):
    """読み込み済み設定から whisper-cli の引数列を作る（純粋関数）。

    -m, -bs, -bo, -tp, -tpi, -et, -lpt, -nth, -mc, -l を並べ、
    prompt が空でなければ --prompt を足す。-ojf / -np / -f / -of は
    呼び出し側（analyze）が付ける。

    -mc は whisper.cpp の n_max_text_ctx。0 にすると initial_prompt が
    デコーダに渡らないので、whisper.yaml の 16384 をそのまま渡す。
    no_context=true は CLI では 1 回の呼び出しごとに文脈が独立している
    という意味なので、-mc とは別で追加の引数は要らない。
    """
    model_path = str(model) if model else config["model"]
    arguments = [
        "-m", model_path,
        "-bs", _format_number(config["beam_size"]),
        "-bo", _format_number(config["best_of"]),
        "-tp", _format_number(config["temperature"]),
        "-tpi", _format_number(config["temperature_inc"]),
        "-et", _format_number(config["entropy_thold"]),
        "-lpt", _format_number(config["logprob_thold"]),
        "-nth", _format_number(config["no_speech_thold"]),
        "-mc", _format_number(config["n_max_text_ctx"]),
        "-l", str(lang),
    ]
    if prompt:
        arguments += ["--prompt", str(prompt)]
    return arguments


def whisper_args_from_yaml(path, lang, prompt, model=None, strict=False):
    """whisper.yaml を読んで whisper-cli の引数列を作る。"""
    config = load_whisper_config(path, strict=strict)
    return whisper_args_from_config(config, lang, prompt, model=model)


def parse_whisper_json(payload):
    """whisper.cpp の JSON full 出力から (text, mean_logprob, n_tokens) を出す。

    transcription[].text を連結し、transcription[].tokens[] のうち特殊トークン
    （text が "[_" で始まるもの）を除いた p の自然対数の平均を取る。
    """
    if not isinstance(payload, dict):
        raise TypeError("whisper JSON は dict である必要がある")
    transcription = payload.get("transcription") or []
    texts = []
    logprobs = []
    for segment in transcription:
        if not isinstance(segment, dict):
            continue
        texts.append(str(segment.get("text", "")))
        for token in segment.get("tokens") or []:
            if not isinstance(token, dict):
                continue
            token_text = str(token.get("text", ""))
            if token_text.lstrip().startswith("[_"):
                continue
            try:
                probability = float(token.get("p"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(probability) or probability < 0.0:
                continue
            if probability == 0.0:
                logprobs.append(float("-inf"))
            else:
                logprobs.append(math.log(probability))
    text = "".join(texts).strip()
    mean_logprob = (sum(logprobs) / len(logprobs)) if logprobs else None
    return text, mean_logprob, len(logprobs)


def prompt_for(kind, lang, hri_params=None, ja_prompt="", strict=False):
    """試行の initial_prompt。日本語は --ja-prompt（既定は空）。"""
    if str(lang).lower() == "ja":
        return ja_prompt or ""
    name_prompt, drink_prompt = english_prompts(hri_params, strict=strict)
    return name_prompt if str(kind).lower() == "name" else drink_prompt


def cache_key(wav_bytes, arguments):
    """(発話 WAV の sha1, 引数) からキャッシュ用のキーを作る。"""
    audio_hash = hashlib.sha1(bytes(wav_bytes)).hexdigest()
    argument_hash = hashlib.sha1(
        json.dumps(list(arguments), ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"{audio_hash}-{argument_hash}"
