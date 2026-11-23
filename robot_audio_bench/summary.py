"""trials.csv の集計・θ_turn・ROC・図。"""
from __future__ import annotations

import csv
import math
from pathlib import Path

SUMMARY_COLUMNS = [
    "lang", "distance_m", "noise", "theta_deg", "n", "n_snr_invalid",
    "snr_median", "snr_q1", "snr_q3",
    "accuracy_pct", "accuracy_ci_low", "accuracy_ci_high",
    "wer_mean", "cer_mean",
]
THETA_TURN_COLUMNS = [
    "lang", "distance_m", "noise", "theta_turn_deg", "baseline_accuracy_pct",
]
ROC_COLUMNS = ["group", "threshold", "fpr", "tpr", "auc", "excluded"]
Z_95 = 1.959963984540054


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_bool(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in ("true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n"):
        return False
    return None


def load_trials(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def wilson_interval(successes, count, z=Z_95):
    """Wilson の 95 % 信頼区間（比率 0..1）。count=0 なら (None, None)。"""
    if count <= 0:
        return None, None
    proportion = successes / float(count)
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * count)) / count) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def percentile(values, fraction):
    """線形補間の分位点（fraction は 0..1）。空なら None。"""
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


def _group_key(row):
    return (
        str(row.get("lang", "")),
        parse_float(row.get("distance_m")),
        str(row.get("noise", "")),
        parse_float(row.get("theta_deg")),
    )


def compute_summary(rows):
    """(lang, distance_m, noise, theta_deg) ごとの集計行を返す。"""
    groups = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
    summary = []
    for (lang, distance, noise, theta), members in groups.items():
        snr_values = []
        snr_invalid = 0
        for row in members:
            valid = parse_bool(row.get("snr_valid"))
            value = parse_float(row.get("snr_db"))
            if valid is False:
                # P_sn <= P_n の試行は「最悪」として -inf を順位に入れる。
                snr_values.append(float("-inf"))
                snr_invalid += 1
            elif value is not None:
                snr_values.append(value)
        wer_values = [value for value in (parse_float(row.get("wer")) for row in members)
                      if value is not None]
        cer_values = [value for value in (parse_float(row.get("cer")) for row in members)
                      if value is not None]
        correct_values = [value for value in (parse_bool(row.get("correct")) for row in members)
                          if value is not None]
        successes = sum(1 for value in correct_values if value)
        if correct_values:
            accuracy = successes / float(len(correct_values))
            low, high = wilson_interval(successes, len(correct_values))
        else:
            accuracy = low = high = None
        summary.append({
            "lang": lang,
            "distance_m": distance,
            "noise": noise,
            "theta_deg": theta,
            "n": len(members),
            "n_snr_invalid": snr_invalid,
            "snr_median": percentile(snr_values, 0.5),
            "snr_q1": percentile(snr_values, 0.25),
            "snr_q3": percentile(snr_values, 0.75),
            "accuracy_pct": None if accuracy is None else accuracy * 100.0,
            "accuracy_ci_low": None if low is None else low * 100.0,
            "accuracy_ci_high": None if high is None else high * 100.0,
            "wer_mean": (sum(wer_values) / len(wer_values)) if wer_values else None,
            "cer_mean": (sum(cer_values) / len(cer_values)) if cer_values else None,
        })
    summary.sort(key=lambda item: (
        item["lang"], -1.0 if item["distance_m"] is None else item["distance_m"],
        item["noise"], -1.0 if item["theta_deg"] is None else item["theta_deg"]))
    return summary


summarize = compute_summary


def _accuracy(members):
    values = [value for value in (parse_bool(row.get("correct")) for row in members)
              if value is not None]
    if not values:
        return None
    return sum(1 for value in values if value) / float(len(values))


def compute_theta_turn(rows, drop_pt=10.0):
    """(lang, distance_m, noise) ごとに、0° から drop_pt 以上落ちる最小 |θ|。

    |θ| の昇順に見る。±90° はどちらか一方でも落ちたら落ちたとみなす
    （同じ |θ| の符号違いの正解率の最小値を使う）。
    """
    groups = {}
    for row in rows:
        lang, distance, noise, theta = _group_key(row)
        if theta is None:
            continue
        groups.setdefault((lang, distance, noise), {}).setdefault(theta, []).append(row)
    results = []
    for (lang, distance, noise), by_theta in groups.items():
        baseline_members = by_theta.get(0.0)
        baseline = _accuracy(baseline_members) if baseline_members else None
        turn_angle = None
        if baseline is not None:
            by_absolute = {}
            for theta, members in by_theta.items():
                if theta == 0.0:
                    continue
                accuracy = _accuracy(members)
                if accuracy is None:
                    continue
                absolute = abs(theta)
                by_absolute.setdefault(absolute, []).append(accuracy)
            for absolute in sorted(by_absolute):
                drop = (baseline - min(by_absolute[absolute])) * 100.0
                if drop >= float(drop_pt) - 1e-9:
                    turn_angle = absolute
                    break
        results.append({
            "lang": lang,
            "distance_m": distance,
            "noise": noise,
            "theta_turn_deg": turn_angle,
            "baseline_accuracy_pct": None if baseline is None else baseline * 100.0,
        })
    results.sort(key=lambda item: (
        item["lang"], -1.0 if item["distance_m"] is None else item["distance_m"],
        item["noise"]))
    return results


theta_turn = compute_theta_turn


def _roc_score(value):
    """ROC 用のスコア。-inf は最下位として残し、空・NaN は None。"""
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if math.isnan(number) else number


def _roc_points(pairs):
    if not pairs:
        return [], None
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    thresholds = [float("inf")]
    for score in sorted({score for score, _ in pairs}, reverse=True):
        if score not in thresholds:
            thresholds.append(score)
    if float("-inf") not in thresholds:
        thresholds.append(float("-inf"))
    points = []
    for threshold in thresholds:
        true_positive = false_positive = 0
        for score, label in pairs:
            predicted = score >= threshold
            if label and predicted:
                true_positive += 1
            elif not label and predicted:
                false_positive += 1
        tpr = true_positive / float(positives) if positives else 0.0
        fpr = false_positive / float(negatives) if negatives else 0.0
        points.append((threshold, fpr, tpr))
    if positives == 0 or negatives == 0:
        return points, None  # 全部正解・全部不正解は AUC 空欄
    ordered = sorted(points, key=lambda point: point[1])
    auc = 0.0
    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        current = ordered[index]
        auc += (current[1] - previous[1]) * (current[2] + previous[2]) / 2.0
    return points, auc


def roc_curve_detail(scores, labels):
    """(points, auc, excluded) を返す。excluded は空・NaN のスコア数。"""
    pairs = []
    excluded = 0
    for score, label in zip(scores, labels):
        score_value = _roc_score(score)
        if score_value is None:
            excluded += 1
            continue
        if label is None:
            continue
        pairs.append((score_value, bool(label)))
    points, auc = _roc_points(pairs)
    return points, auc, excluded


def roc_curve(scores, labels):
    """score を信頼度、label=correct として (threshold, fpr, tpr) の列と AUC。

    -inf のスコアは最下位として扱う。空のスコアは除外され、その数は
    roc_curve_detail() / roc_groups() で取れる。
    全部正解・全部不正解のとき AUC は None（CSV では空欄）。
    """
    points, auc, _ = roc_curve_detail(scores, labels)
    return points, auc


def roc_auc(scores, labels):
    return roc_curve(scores, labels)[1]


auc = roc_auc


def roc_groups(rows):
    """ROC の (group, points, auc, excluded) を all と言語ごとに返す。"""
    groups = [("all", rows)]
    for lang in ("en", "ja"):
        members = [row for row in rows if str(row.get("lang", "")) == lang]
        if members:
            groups.append((lang, members))
    results = []
    for name, members in groups:
        scores = [row.get("mean_logprob") for row in members]
        labels = [parse_bool(row.get("correct")) for row in members]
        points, auc, excluded = roc_curve_detail(scores, labels)
        results.append((name, points, auc, excluded))
    return results


def _format(value, digits=4):
    if value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _number(value):
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def write_summary_csv(summary, path):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in summary:
            writer.writerow({
                "lang": row["lang"],
                "distance_m": _number(row["distance_m"]),
                "noise": row["noise"],
                "theta_deg": _number(row["theta_deg"]),
                "n": row["n"],
                "n_snr_invalid": row["n_snr_invalid"],
                "snr_median": _format(row["snr_median"], 3),
                "snr_q1": _format(row["snr_q1"], 3),
                "snr_q3": _format(row["snr_q3"], 3),
                "accuracy_pct": _format(row["accuracy_pct"], 2),
                "accuracy_ci_low": _format(row["accuracy_ci_low"], 2),
                "accuracy_ci_high": _format(row["accuracy_ci_high"], 2),
                "wer_mean": _format(row["wer_mean"], 4),
                "cer_mean": _format(row["cer_mean"], 4),
            })


def write_theta_turn_csv(results, path):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=THETA_TURN_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in results:
            writer.writerow({
                "lang": row["lang"],
                "distance_m": _number(row["distance_m"]),
                "noise": row["noise"],
                "theta_turn_deg": _number(row["theta_turn_deg"]),
                "baseline_accuracy_pct": _format(row["baseline_accuracy_pct"], 2),
            })


def write_roc_csv(groups, path):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for name, points, auc, excluded in groups:
            if not points:
                writer.writerow({"group": name, "threshold": "", "fpr": "",
                                 "tpr": "", "auc": "", "excluded": excluded})
                continue
            for threshold, fpr, tpr in points:
                writer.writerow({
                    "group": name,
                    "threshold": "inf" if threshold == float("inf")
                    else ("-inf" if threshold == float("-inf") else f"{threshold:.6f}"),
                    "fpr": _format(fpr, 6),
                    "tpr": _format(tpr, 6),
                    "auc": _format(auc, 6),
                    "excluded": excluded,
                })


def theta_turn_markdown(results):
    lines = [
        "| lang | distance_m | noise | theta_turn_deg | baseline_accuracy_pct |",
        "|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['lang']} | {_number(row['distance_m']) or '-'} | {row['noise']} "
            f"| {_number(row['theta_turn_deg']) or '-'} "
            f"| {_format(row['baseline_accuracy_pct'], 2) or '-'} |")
    return "\n".join(lines)


def plot_theta(summary, output):
    """θ ごとの SNR 中央値（左）と正解率（右）。言語は行、条件は 4 本の線。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    langs = sorted({row["lang"] for row in summary if row["lang"]})
    if not langs:
        return False
    figure, axes = plt.subplots(len(langs), 2, figsize=(12, 4 * len(langs)), squeeze=False)
    conditions = sorted({
        (row["distance_m"], row["noise"]) for row in summary
        if row["distance_m"] is not None and row["noise"] in ("N1", "N2")
    })
    for row_index, lang in enumerate(langs):
        snr_axis, accuracy_axis = axes[row_index]
        for distance, noise in conditions:
            members = [
                row for row in summary
                if row["lang"] == lang and row["distance_m"] == distance
                and row["noise"] == noise and row["theta_deg"] is not None
            ]
            members.sort(key=lambda item: item["theta_deg"])
            if not members:
                continue
            label = f"d={_number(distance)} m {noise}"
            snr_points = [
                (row["theta_deg"], row["snr_median"]) for row in members
                if row["snr_median"] is not None and math.isfinite(row["snr_median"])
            ]
            if snr_points:
                snr_axis.plot([point[0] for point in snr_points],
                              [point[1] for point in snr_points], marker="o", label=label)
            accuracy_points = [
                (row["theta_deg"], row["accuracy_pct"]) for row in members
                if row["accuracy_pct"] is not None
            ]
            if accuracy_points:
                accuracy_axis.plot([point[0] for point in accuracy_points],
                                   [point[1] for point in accuracy_points],
                                   marker="o", label=label)
        snr_axis.set_title(f"{lang}: SNR median [dB]")
        snr_axis.set_xlabel("theta [deg]")
        snr_axis.set_ylabel("SNR median [dB]")
        snr_axis.grid(True, alpha=0.3)
        snr_axis.legend(fontsize=7)
        accuracy_axis.set_title(f"{lang}: accuracy [%]")
        accuracy_axis.set_xlabel("theta [deg]")
        accuracy_axis.set_ylabel("accuracy [%]")
        accuracy_axis.set_ylim(0, 105)
        accuracy_axis.grid(True, alpha=0.3)
        accuracy_axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return True


def plot_roc(groups, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axis = plt.subplots(figsize=(6, 6))
    plotted = False
    for name, points, auc, excluded in groups:
        if not points or auc is None:
            continue
        axis.plot([point[1] for point in points], [point[2] for point in points],
                  marker=".", label=f"{name} (AUC={auc:.3f})")
        plotted = True
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set_xlabel("FPR")
    axis.set_ylabel("TPR")
    axis.set_title("ROC (confidence = mean_logprob)")
    axis.grid(True, alpha=0.3)
    if plotted:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return True
