"""刺激文（名前・飲み物）と stimuli.csv の生成。"""
from __future__ import annotations

import csv
from pathlib import Path

from .config import (
    JAPANESE_DRINKS, JAPANESE_NAMES, OUT_OF_LIST_NAMES, load_hri_config,
)

STIMULI_COLUMNS = ["utt_id", "lang", "kind", "keyword", "in_list", "text", "wav"]
ENGLISH_NAME_SENTENCES = 20
ENGLISH_DRINK_SENTENCES = 20
JAPANESE_NAME_SENTENCES = 20
JAPANESE_DRINK_SENTENCES = 20


def _cycled(values, count):
    if not values:
        raise ValueError("候補リストが空")
    return [values[index % len(values)] for index in range(count)]


def _row(utt_id, lang, kind, keyword, in_list, text):
    return {
        "utt_id": utt_id,
        "lang": lang,
        "kind": kind,
        "keyword": keyword,
        "in_list": "true" if in_list else "false",
        "text": text,
        "wav": f"{utt_id}_<speaker>.wav",
    }


def build_stimuli(hri_params=None, config=None):
    """英語 45 文（名前 25・飲み物 20）と日本語 40 文（名前 20・飲み物 20）を作る。

    英語の名前 20 文は name_list を先頭から回して全候補を使い、足りない分を
    先頭から繰り返す。リスト外の名前 5 文は固定値（in_list=false）。
    config を渡すと yaml の読み直しを省く（stimuli の警告用）。
    """
    config = config if config is not None else load_hri_config(hri_params)
    names = config["name_list"]
    drinks = config["drink_list"]

    rows = []
    for index, name in enumerate(_cycled(names, ENGLISH_NAME_SENTENCES)):
        keyword = name.capitalize()
        rows.append(_row(f"en_name_{index:02d}", "en", "name", keyword, True,
                         f"My name is {keyword}."))
    for index, name in enumerate(OUT_OF_LIST_NAMES):
        rows.append(_row(f"en_oov_{index:02d}", "en", "name", name, False,
                         f"My name is {name}."))
    for index, drink in enumerate(_cycled(drinks, ENGLISH_DRINK_SENTENCES)):
        rows.append(_row(f"en_drink_{index:02d}", "en", "drink", drink, True,
                         f"My favorite drink is {drink}."))
    for index, name in enumerate(_cycled(JAPANESE_NAMES, JAPANESE_NAME_SENTENCES)):
        rows.append(_row(f"ja_name_{index:02d}", "ja", "name", name, True,
                         f"私の名前は{name}です"))
    for index, drink in enumerate(_cycled(JAPANESE_DRINKS, JAPANESE_DRINK_SENTENCES)):
        rows.append(_row(f"ja_drink_{index:02d}", "ja", "drink", drink, True,
                         f"好きな飲み物は{drink}です"))
    return rows


def write_stimuli_csv(rows, path):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=STIMULI_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_stimuli_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
