"""robot-audio-bench: Robot audio benchmarkの CLI。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyze import run_analysis
from .config import load_hri_config
from .directivity import (
    compute_directivity, load_recordings, write_directivity_csv,
)
from .noise import make_babble, make_pink
from .playlist import build_playlist, parse_speakers
from .stimuli import build_stimuli, read_stimuli_csv, write_stimuli_csv
from .summary import (
    compute_summary, compute_theta_turn, load_trials, plot_roc, plot_theta,
    roc_groups, theta_turn_markdown, write_roc_csv, write_summary_csv,
    write_theta_turn_csv,
)


def _output_directory(arguments):
    directory = Path(getattr(arguments, "out_dir", "."))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def command_stimuli(arguments):
    config = load_hri_config(arguments.hri_params)
    if not config["loaded"]:
        print(
            f"stimuli: 警告: hri_params.yaml を読めないので既定値を使う: {config['error']}",
            file=sys.stderr)
    rows = build_stimuli(config=config)
    write_stimuli_csv(rows, arguments.output)
    print(f"stimuli: {len(rows)} 文 -> {arguments.output}")
    return 0


def command_playlist(arguments):
    stimuli_path = arguments.stimuli_option or arguments.stimuli
    if stimuli_path is None:
        raise ValueError("stimuli.csv を指定する（位置引数か --stimuli）")
    stimuli = read_stimuli_csv(stimuli_path)
    out_dir = _output_directory(arguments)
    wav_path = out_dir / f"playlist_{arguments.lang}.wav"
    csv_path = out_dir / f"playlist_{arguments.lang}.csv"
    rows = build_playlist(
        stimuli, arguments.wav_dir, parse_speakers(arguments.speakers),
        arguments.lang, seed=arguments.seed,
        output_wav=wav_path, output_csv=csv_path)
    print(f"playlist: {len(rows)} 試行 -> {wav_path}, {csv_path}")
    return 0


def command_noise_babble(arguments):
    info = make_babble(
        arguments.librispeech, speakers=arguments.speakers,
        duration=arguments.duration, seed=arguments.seed, output=arguments.output)
    print(f"noise babble: {info['speaker_count']} 話者, {info['duration_sec']:.1f} s -> {arguments.output}")
    return 0


def command_noise_pink(arguments):
    info = make_pink(
        duration=arguments.duration, seed=arguments.seed, output=arguments.output)
    print(f"noise pink: {info['duration_sec']:.1f} s -> {arguments.output}")
    return 0


def command_analyze(arguments):
    rows, offset = run_analysis(
        arguments.recording, arguments.playlist, arguments.out,
        arguments.theta_deg, arguments.distance_m, arguments.noise, arguments.lang,
        yaw_deg=arguments.yaw_deg, offset_s=arguments.offset_s,
        whisper_cli=arguments.whisper_cli, model=arguments.model,
        whisper_yaml=arguments.whisper_yaml, hri_params=arguments.hri_params,
        ja_prompt=arguments.ja_prompt, asr_cache_dir=arguments.asr_cache_dir,
        no_asr=arguments.no_asr, append=arguments.append)
    print(f"analyze: {len(rows)} 試行, offset={offset:.6f} s -> {arguments.out}")
    return 0


def command_summary(arguments):
    trials = load_trials(arguments.trials)
    if not trials:
        raise ValueError("trials.csv が空")
    summary = compute_summary(trials)
    turns = compute_theta_turn(trials, drop_pt=arguments.drop_pt)
    groups = roc_groups(trials)
    out_dir = _output_directory(arguments)
    write_summary_csv(summary, out_dir / "summary.csv")
    write_theta_turn_csv(turns, out_dir / "theta_turn.csv")
    write_roc_csv(groups, out_dir / "roc.csv")
    print(theta_turn_markdown(turns))
    for name, _points, _auc, excluded in groups:
        print(f"ROC {name}: excluded(empty mean_logprob)={excluded}")
    print(f"\nsummary: {len(summary)} 条件 -> {out_dir}")
    if not arguments.no_plot:
        if not plot_theta(summary, out_dir / "fig_theta.png"):
            print("summary: 警告: matplotlib が無いので fig_theta.png を飛ばす", file=sys.stderr)
        if not plot_roc(groups, out_dir / "roc.png"):
            print("summary: 警告: matplotlib が無いので roc.png を飛ばす", file=sys.stderr)
    return 0


def command_directivity(arguments):
    recordings = load_recordings(arguments.rec)
    rows = compute_directivity(
        recordings, start_s=arguments.start_s, duration_s=arguments.duration_s,
        reference_deg=arguments.reference_deg)
    write_directivity_csv(rows, arguments.output)
    print(f"directivity: {len(recordings)} 角度, {len(rows)} 行 -> {arguments.output}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="robot-audio-bench",
        description="Robot audio benchmark: 刺激・再生リスト・雑音・解析・集計")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stimuli = subparsers.add_parser("stimuli", help="stimuli.csv を作る")
    stimuli.add_argument("--dialogue-config", "--hri-params", dest="hri_params", type=Path, default=None,
                        help="既定: パッケージ同梱の設定")
    stimuli.add_argument("--output", type=Path, default=Path("stimuli.csv"))
    stimuli.set_defaults(function=command_stimuli)

    playlist = subparsers.add_parser("playlist", help="試行順の playlist WAV / CSV を作る")
    playlist.add_argument("stimuli", nargs="?", type=Path)
    playlist.add_argument("--stimuli", dest="stimuli_option", type=Path, default=None)
    playlist.add_argument("--wav-dir", "--recordings-dir", "--recordings", dest="wav_dir",
                          type=Path, required=True)
    playlist.add_argument("--speakers", required=True, help="例: alice,bob")
    playlist.add_argument("--lang", choices=("en", "ja"), required=True)
    playlist.add_argument("--seed", type=int, default=0)
    playlist.add_argument("--out-dir", "--output-dir", dest="out_dir", type=Path, default=Path("."))
    playlist.set_defaults(function=command_playlist)

    noise = subparsers.add_parser("noise", help="バブル雑音・ピンクノイズを作る")
    noise_subparsers = noise.add_subparsers(dest="noise_command", required=True)

    babble = noise_subparsers.add_parser("babble", help="LibriSpeech からバブル雑音")
    babble.add_argument("--librispeech", type=Path, required=True)
    babble.add_argument("--speakers", type=int, default=8)
    babble.add_argument("--duration", type=float, default=600.0)
    babble.add_argument("--seed", type=int, default=0)
    babble.add_argument("--output", type=Path, default=Path("babble.wav"))
    babble.set_defaults(function=command_noise_babble)

    pink = noise_subparsers.add_parser("pink", help="ピンクノイズ")
    pink.add_argument("--duration", type=float, default=10.0)
    pink.add_argument("--seed", type=int, default=0)
    pink.add_argument("--output", type=Path, default=Path("pink.wav"))
    pink.set_defaults(function=command_noise_pink)

    analyze = subparsers.add_parser("analyze", help="録音から SNR と認識結果を出す")
    analyze.add_argument("--recording", type=Path, required=True)
    analyze.add_argument("--playlist", type=Path, required=True)
    analyze.add_argument("--theta-deg", type=float, required=True)
    analyze.add_argument("--yaw-deg", type=float, default=None)
    analyze.add_argument("--distance-m", type=float, required=True)
    analyze.add_argument("--noise", choices=("N0", "N1", "N2"), required=True)
    analyze.add_argument("--lang", choices=("en", "ja"), required=True)
    analyze.add_argument("--offset-s", type=float, default=None,
                         help="省略時は録音先頭 30 s から 1 kHz トーンを探す")
    analyze.add_argument("--out", type=Path, required=True)
    analyze.add_argument("--whisper-cli", default="whisper-cli")
    analyze.add_argument("--model", default=None)
    analyze.add_argument("--whisper-yaml", type=Path, default=None,
                         help="既定: パッケージ同梱の設定")
    analyze.add_argument("--dialogue-config", "--hri-params", dest="hri_params", type=Path, default=None,
                         help="既定: パッケージ同梱の設定")
    analyze.add_argument("--ja-prompt", default="")
    analyze.add_argument("--asr-cache-dir", type=Path, default=None)
    analyze.add_argument("--no-asr", action="store_true")
    analyze.add_argument("--append", action="store_true")
    analyze.set_defaults(function=command_analyze)

    summary = subparsers.add_parser("summary", help="集計・θ_turn・ROC・図")
    summary.add_argument("trials", nargs="+", type=Path)
    summary.add_argument("--drop-pt", type=float, default=10.0)
    summary.add_argument("--out-dir", "--output-dir", dest="out_dir", type=Path, default=Path("."))
    summary.add_argument("--no-plot", action="store_true")
    summary.set_defaults(function=command_summary)

    directivity = subparsers.add_parser("directivity", help="ピンクノイズの指向性")
    directivity.add_argument("--rec", action="append", required=True,
                             help="THETA=PATH（例: --rec 0=rec0.wav --rec 90=rec90.wav）")
    directivity.add_argument("--start-s", type=float, default=0.0)
    directivity.add_argument("--duration-s", type=float, default=10.0)
    directivity.add_argument("--reference-deg", type=float, default=0.0)
    directivity.add_argument("--output", type=Path, default=Path("directivity.csv"))
    directivity.set_defaults(function=command_directivity)
    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.function(arguments))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"robot-audio-bench: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
