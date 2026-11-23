"""WER / CER と keyword 照合（標準ライブラリの編集距離だけ）。"""
from __future__ import annotations

import re

# 句読点（英語・日本語）と括弧・記号。正規化で落とす。
PUNCTUATION = ".,!?;:\"'`()[]{}<>、。！？「」『』【】・…—–-"
_PUNCTUATION_TABLE = str.maketrans({character: "" for character in PUNCTUATION})


def normalize_text(text, lang="en"):
    """小文字化・句読点除去。英語は空白で単語分割、日本語は空白も除去。"""
    normalized = str(text or "").lower().translate(_PUNCTUATION_TABLE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if str(lang).lower() == "ja":
        normalized = normalized.replace(" ", "")
    return normalized


def _tokens(text, lang="en"):
    normalized = normalize_text(text, lang)
    if str(lang).lower() == "ja":
        return list(normalized)  # 日本語は文字単位
    return normalized.split() if normalized else []


def edit_distance(reference, hypothesis):
    """レーベンシュタイン距離（置換・挿入・削除それぞれ 1）。"""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(
                previous[column] + 1,             # 削除
                current[column - 1] + 1,          # 挿入
                previous[column - 1] + (ref_item != hyp_item),  # 置換
            ))
        previous = current
    return previous[-1]


def word_error_rate(reference, hypothesis):
    """WER = 編集距離 / 参照の単語数。参照が空なら 0（仮説も空）か 1。"""
    ref_tokens = _tokens(reference, "en")
    hyp_tokens = _tokens(hypothesis, "en")
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return edit_distance(ref_tokens, hyp_tokens) / float(len(ref_tokens))


def character_error_rate(reference, hypothesis):
    """CER = 編集距離 / 参照の文字数。参照が空なら 0（仮説も空）か 1。"""
    ref_characters = _tokens(reference, "ja")
    hyp_characters = _tokens(hypothesis, "ja")
    if not ref_characters:
        return 0.0 if not hyp_characters else 1.0
    return edit_distance(ref_characters, hyp_characters) / float(len(ref_characters))


def is_correct(keyword, hypothesis, lang="en"):
    """task_manager.py の _listen_for_keyword と同じ照合。

    英語: re.search(r"\\b" + re.escape(keyword.lower()) + r"\\b", text.lower())。
          単語境界で照合するので "Sarah" は "sara" に一致しない。
    日本語: 単語境界が無いので、正規化した認識文に keyword（カタカナ）が
          部分文字列として含まれるかで判定する。
    """
    text = str(hypothesis or "")
    keyword = str(keyword or "")
    if str(lang).lower() == "ja":
        return normalize_text(keyword, "ja") in normalize_text(text, "ja") if keyword else False
    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


def infer_kind(utt_id, text=""):
    """playlist の utt_id / 参照文から kind を推定する。

    playlist CSV には kind 列が無いので、`..._drink_...` か参照文の
    "drink" / "飲み物" を見る。どちらでもなければ name。
    """
    haystack = f"{utt_id} {text}".lower()
    return "drink" if ("drink" in haystack or "飲み物" in haystack) else "name"


def infer_in_list(utt_id):
    """playlist の utt_id から in_list を推定する（..._oov_... はリスト外）。"""
    return "oov" not in str(utt_id).lower()


# 短い別名。
wer = word_error_rate
cer = character_error_rate
keyword_matches = is_correct
