"""刺激・ASR プロンプトで使う候補語と hri_params.yaml の読み込み。

yaml は関数内で import する。読めないときは task_manager.py の
_NAME_LIST / _DRINK_LIST と同じ値を定数として使う。
"""
from __future__ import annotations

from pathlib import Path

# task_manager.py の _NAME_LIST / _DRINK_LIST と同じ値（yaml が読めないときの既定）。
FALLBACK_NAME_LIST = [
    "sophie", "kevin", "julia", "gabrielle", "emma",
    "robin", "noah", "sara", "john", "harrie",
    "laura", "liam", "peter", "lucas", "susan", "william",
]
FALLBACK_DRINK_LIST = [
    "espresso", "pepsi", "milk", "tea", "coffee",
    "orange juice", "strawberry juice", "lemon juice", "apple juice",
]

# リスト外の名前（英語刺激の 5 文）。候補リストと重ならない固定値。
OUT_OF_LIST_NAMES = ["Oliver", "Hannah", "Mateo", "Chloe", "Ethan"]

# 日本語刺激の固定リスト（カタカナ、20 個ずつ）。
# 「ジュース」「水」のような短い語は、ほかの語の部分文字列になって
# is_correct の部分一致で誤って正解になるため使わない。
JAPANESE_NAMES = [
    "ソフィー", "ケビン", "ジュリア", "ガブリエル", "エマ",
    "ロビン", "ノア", "サラ", "ジョン", "ハリー",
    "ローラ", "マックス", "ピーター", "ルーカス", "スーザン",
    "ウィリアム", "オリバー", "ハンナ", "マテオ", "クロエ",
]
JAPANESE_DRINKS = [
    "コーヒー", "紅茶", "牛乳", "エスプレッソ", "ペプシ",
    "オレンジジュース", "アップルジュース", "レモンジュース",
    "いちごジュース", "ミネラルウォーター", "緑茶", "烏龍茶",
    "コーラ", "ぶどうジュース", "ワイン", "ビール", "炭酸水",
    "アイスティー", "ソーダ", "ミルクティー",
]

DEFAULT_HRI_PARAMS = Path(__file__).resolve().parent / "config" / "dialogue.yaml"


class ConfigurationError(RuntimeError):
    """解析に必要な設定ファイルが無い・読めないときのエラー。"""


def resolve_repository_file(relative):
    """既定の設定ファイルを作業ディレクトリ→リポジトリルートの順で探す。

    1. 現在の作業ディレクトリ基準の `relative`
    2. このファイル (`config.py`) から親をたどり、`relative` が存在するディレクトリ

    見つからなければ None。設定ファイルの既定パス解決はここに集約する。
    """
    relative = Path(relative)
    cwd_candidate = Path.cwd() / relative
    if cwd_candidate.is_file():
        return cwd_candidate
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    return None


def _candidate_path(path, default_relative):
    """明示パスならそれを、既定値（None / 既定の相対パス）なら探索結果を使う。"""
    if path is None:
        return resolve_repository_file(default_relative)
    candidate = Path(path)
    if not candidate.is_absolute() and candidate == default_relative:
        return resolve_repository_file(default_relative)
    return candidate


def _as_string_list(value, fallback):
    if isinstance(value, (list, tuple)) and value:
        result = [str(item) for item in value if str(item)]
        if result:
            return result
    return list(fallback)


def _fallback_config(error):
    return {
        "name_list": list(FALLBACK_NAME_LIST),
        "drink_list": list(FALLBACK_DRINK_LIST),
        "name_prompt": "",
        "drink_prompt": "",
        "loaded": False,
        "source": "",
        "error": str(error),
    }


def load_hri_config(path=None, strict=False):
    """hri_params.yaml から name_list / drink_list / *_prompt を読む。

    `path` が None（既定）なら resolve_repository_file() で探す。
    strict=True で見つからない・読めないときは ConfigurationError。
    strict=False（stimuli 用）では task_manager.py と同じフォールバック値を使い、
    戻り値の loaded / error で失敗を伝える。
    """
    candidate = _candidate_path(path, DEFAULT_HRI_PARAMS)
    try:
        if candidate is None or not candidate.is_file():
            raise FileNotFoundError(
                f"hri_params.yaml が見つからない: {candidate or DEFAULT_HRI_PARAMS}")
        import yaml  # 遅延 import（import 時に yaml を読まない）
        with candidate.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        parameters = ((data.get("hri_task_manager") or {}).get("ros__parameters") or {})
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(
                "hri_params.yaml に hri_task_manager.ros__parameters が無い")
    except Exception as error:
        if strict:
            shown = candidate if candidate is not None else DEFAULT_HRI_PARAMS
            raise ConfigurationError(
                f"hri_params.yaml を読めない: {shown} ({error})") from error
        return _fallback_config(error)

    name_prompt = parameters.get("name_prompt", "")
    drink_prompt = parameters.get("drink_prompt", "")
    return {
        "name_list": _as_string_list(parameters.get("name_list"), FALLBACK_NAME_LIST),
        "drink_list": _as_string_list(parameters.get("drink_list"), FALLBACK_DRINK_LIST),
        "name_prompt": name_prompt if isinstance(name_prompt, str) else "",
        "drink_prompt": drink_prompt if isinstance(drink_prompt, str) else "",
        "loaded": True,
        "source": str(candidate),
        "error": "",
    }


def build_name_prompt(name_list):
    """task_manager.py の _build_name_prompt と同じ規則。"""
    return (
        "The guest says their name. Possible names: "
        + ", ".join(name.capitalize() for name in name_list)
        + "."
    )


def build_drink_prompt(drink_list):
    """task_manager.py の _build_drink_prompt と同じ規則。"""
    return (
        "The guest says their favorite drink. Possible drinks: "
        + ", ".join(drink_list)
        + "."
    )


def english_prompts(path=None, strict=False):
    """英語 ASR の (name_prompt, drink_prompt)。

    yaml の name_prompt / drink_prompt が空文字なら、候補リストから
    task_manager.py と同じ規則で組み立てる。
    """
    config = load_hri_config(path, strict=strict)
    name_prompt = config["name_prompt"] or build_name_prompt(config["name_list"])
    drink_prompt = config["drink_prompt"] or build_drink_prompt(config["drink_list"])
    return name_prompt, drink_prompt
