"""Hear-and-Face ベンチ（Issue #261）の試験。ROS 不要。"""
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import random
import tempfile
import unittest
import wave
from pathlib import Path

from robot_audio_bench.analyze import (
    TRIAL_COLUMNS, analyze_recording, write_trials_csv)
from robot_audio_bench.cli import main
from robot_audio_bench.config import (
    DEFAULT_HRI_PARAMS, FALLBACK_DRINK_LIST, FALLBACK_NAME_LIST,
    JAPANESE_DRINKS, JAPANESE_NAMES, OUT_OF_LIST_NAMES,
    build_drink_prompt, build_name_prompt, english_prompts, load_hri_config,
    resolve_repository_file,
)
from robot_audio_bench.directivity import (
    THIRD_OCTAVE_CENTERS, compute_directivity, supercardioid_db,
)
from robot_audio_bench.playlist import build_playlist
from robot_audio_bench.signal import (
    detect_tone_start, snr_from_segments,
)
from robot_audio_bench.stimuli import build_stimuli, write_stimuli_csv
from robot_audio_bench.summary import (
    compute_summary, compute_theta_turn, roc_curve, roc_curve_detail,
    roc_groups, wilson_interval, write_summary_csv,
)
from robot_audio_bench.text_metrics import (
    character_error_rate, is_correct, normalize_text, word_error_rate,
)
from robot_audio_bench.wav import (
    SAMPLE_RATE, read_wav, tone, wav_info, write_wav,
)
from robot_audio_bench.whisper import (
    DEFAULT_WHISPER_YAML, load_whisper_config, parse_whisper_json,
    whisper_args_from_yaml,
)

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
REPO_ROOT = Path(__file__).resolve().parents[1]
WHISPER_YAML = DEFAULT_WHISPER_YAML


def write_sine(path, frequency=440.0, seconds=0.3, amplitude=0.25):
    write_wav(path, tone(frequency, seconds, amplitude=amplitude))
    return path


def random_noise(count, amplitude, seed=0):
    generator = random.Random(seed)
    return [generator.uniform(-amplitude, amplitude) for _ in range(count)]


def trial_row(lang, distance, noise, theta, correct, logprob, snr=10.0,
              snr_valid=True):
    return {
        "trial_idx": 0, "utt_id": "en_name_00", "speaker": "a", "lang": lang,
        "kind": "name", "keyword": "Sophie", "in_list": "true",
        "theta_deg": str(theta), "yaw_deg": "", "distance_m": str(distance),
        "noise": noise, "snr_db": str(snr),
        "snr_valid": "true" if snr_valid else "false",
        "noise_dbfs": "-40", "speech_dbfs": "-30", "hyp": "",
        "ref": "My name is Sophie.", "prompt": "", "wer": "0", "cer": "",
        "correct": "true" if correct else "false", "mean_logprob": str(logprob),
        "n_tokens": "3", "offset_s": "0", "recording": "rec.wav",
    }


class StimuliTest(unittest.TestCase):
    def test_english_and_japanese_counts(self):
        rows = build_stimuli(REPO_ROOT / "does-not-exist.yaml")
        english = [row for row in rows if row["lang"] == "en"]
        japanese = [row for row in rows if row["lang"] == "ja"]
        self.assertEqual(len(english), 45)
        self.assertEqual(len(japanese), 40)
        self.assertEqual(len([row for row in english if row["kind"] == "name"]), 25)
        self.assertEqual(len([row for row in english if row["kind"] == "drink"]), 20)
        self.assertEqual(len([row for row in japanese if row["kind"] == "name"]), 20)
        self.assertEqual(len([row for row in japanese if row["kind"] == "drink"]), 20)

    def test_out_of_list_names_do_not_overlap_the_lists(self):
        listed = {name.lower() for name in FALLBACK_NAME_LIST}
        for name in OUT_OF_LIST_NAMES:
            self.assertNotIn(name.lower(), listed)
        rows = build_stimuli(REPO_ROOT / "does-not-exist.yaml")
        out_of_list = [row for row in rows if row["in_list"] == "false"]
        self.assertEqual(len(out_of_list), 5)
        self.assertTrue(all(row["kind"] == "name" for row in out_of_list))

    def test_english_names_use_every_list_entry(self):
        rows = build_stimuli(REPO_ROOT / "does-not-exist.yaml")
        keywords = {row["keyword"].lower() for row in rows
                    if row["utt_id"].startswith("en_name_")}
        self.assertEqual(keywords, {name.lower() for name in FALLBACK_NAME_LIST})

    def test_japanese_keywords_are_not_substrings_of_each_other(self):
        # 英語は単語境界で照合するので、liam/william, tea/iced tea のような
        # 部分文字列は問題にならない。日本語は部分一致なのでリスト内で調べる。
        for values in (JAPANESE_NAMES, JAPANESE_DRINKS):
            lowered = [value.lower() for value in values]
            for index, keyword in enumerate(lowered):
                for other_index, other in enumerate(lowered):
                    if index != other_index:
                        self.assertNotIn(keyword, other,
                                         f"{values[index]!r} は {values[other_index]!r} の部分文字列")

    def test_stimuli_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stimuli.csv"
            write_stimuli_csv(build_stimuli(REPO_ROOT / "does-not-exist.yaml"), path)
            with path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 85)
            self.assertTrue(rows[0]["wav"].endswith("_<speaker>.wav"))


class ConfigPathTest(unittest.TestCase):
    def test_default_paths_resolve_from_a_temporary_cwd(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                self.assertIsNotNone(resolve_repository_file(DEFAULT_HRI_PARAMS))
                hri = load_hri_config()
                self.assertTrue(hri["loaded"])
                self.assertIn("sophie", hri["name_list"])
                self.assertIn("Sophie", english_prompts()[0])
                whisper = load_whisper_config()
                self.assertTrue(whisper["loaded"])
                self.assertEqual(whisper["model"], "ggml-large-v3-turbo-q5_0.bin")
            finally:
                os.chdir(old_cwd)

    def test_strict_missing_config_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.yaml"
            with self.assertRaises(RuntimeError):
                load_hri_config(missing, strict=True)
            with self.assertRaises(RuntimeError):
                load_whisper_config(missing, strict=True)

    def test_stimuli_warns_when_it_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = main(["stimuli", "--hri-params", str(directory / "missing.yaml"),
                               "--output", str(directory / "stimuli.csv")])
            self.assertEqual(result, 0)
            self.assertIn("警告", stderr.getvalue())
            self.assertIn("既定値", stderr.getvalue())


class PlaylistTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.stimuli = build_stimuli(REPO_ROOT / "does-not-exist.yaml")[:2]
        for row in self.stimuli:
            write_sine(self.root / f"{row['utt_id']}_a.wav", seconds=0.3)

    def test_timeline_and_wav_length(self):
        wav_path = self.root / "playlist_en.wav"
        csv_path = self.root / "playlist_en.csv"
        rows = build_playlist(self.stimuli, self.root, ["a"], "en", seed=3,
                              output_wav=wav_path, output_csv=csv_path)
        self.assertEqual(rows[0]["noise_start_s"], 1.5)
        self.assertEqual(rows[0]["noise_end_s"], 3.5)
        self.assertEqual(rows[0]["speech_start_s"], 3.5)
        self.assertEqual(rows[0]["speech_end_s"], 3.8)
        self.assertAlmostEqual(rows[1]["noise_start_s"], 3.8 + 1.0)
        frames, rate, channels, width = wav_info(wav_path)
        self.assertEqual((rate, channels, width), (SAMPLE_RATE, 1, 2))
        self.assertAlmostEqual(frames / rate, rows[-1]["speech_end_s"])
        with csv_path.open(encoding="utf-8") as stream:
            written = list(csv.DictReader(stream))
        self.assertEqual(len(written), 2)
        self.assertAlmostEqual(float(written[-1]["speech_end_s"]), frames / rate)

    def test_seed_changes_order_but_keeps_the_set(self):
        first = build_playlist(self.stimuli, self.root, ["a"], "en", seed=1)
        second = build_playlist(self.stimuli, self.root, ["a"], "en", seed=2)
        self.assertEqual(len(first), len(second))
        self.assertEqual({row["utt_id"] for row in first},
                         {row["utt_id"] for row in second})


class SyncTest(unittest.TestCase):
    def test_tone_start_is_found_within_20_ms(self):
        offset = 1.234
        noise = random_noise(int(3.0 * SAMPLE_RATE), 100.0, seed=4)
        start = int(round(offset * SAMPLE_RATE))
        for index, value in enumerate(tone(1000.0, 0.5, amplitude=0.5)):
            noise[start + index] += value
        detected = detect_tone_start(noise)
        self.assertIsNotNone(detected)
        self.assertLessEqual(abs(detected - offset), 0.02)

    def test_tone_starting_just_after_recording_start_is_found(self):
        offset = 0.1
        noise = random_noise(int(3.0 * SAMPLE_RATE), 100.0, seed=8)
        start = int(round(offset * SAMPLE_RATE))
        for index, value in enumerate(tone(1000.0, 0.5, amplitude=0.5)):
            noise[start + index] += value
        detected = detect_tone_start(noise)
        self.assertIsNotNone(detected)
        self.assertLessEqual(abs(detected - offset), 0.02)

    def test_no_tone_returns_none(self):
        noise = random_noise(int(2.0 * SAMPLE_RATE), 100.0, seed=5)
        self.assertIsNone(detect_tone_start(noise))


class SnrTest(unittest.TestCase):
    def test_known_sine_and_noise(self):
        amplitude = 4000.0
        noise_amplitude = 300.0
        signal = [amplitude * math.sin(2.0 * math.pi * 440.0 * index / SAMPLE_RATE)
                  for index in range(SAMPLE_RATE)]
        noise = random_noise(SAMPLE_RATE, noise_amplitude, seed=6)
        speech = [value + disturbance for value, disturbance in zip(signal, noise)]
        expected = 10.0 * math.log10(
            (amplitude * amplitude / 2.0) / (noise_amplitude * noise_amplitude / 3.0))
        snr, valid, _, _ = snr_from_segments(noise, speech)
        self.assertTrue(valid)
        self.assertAlmostEqual(snr, expected, delta=0.5)

    def test_zero_noise_and_no_signal_are_invalid(self):
        snr, valid, noise_dbfs, _ = snr_from_segments([0] * 100, [100] * 100)
        self.assertIsNone(snr)
        self.assertFalse(valid)
        self.assertIsNone(noise_dbfs)
        snr, valid, _, _ = snr_from_segments([100] * 100, [100] * 100)
        self.assertIsNone(snr)
        self.assertFalse(valid)


class TextMetricsTest(unittest.TestCase):
    def test_wer_for_substitution_insertion_and_deletion(self):
        self.assertAlmostEqual(word_error_rate("my name is sara", "my name is sarah"), 0.25)
        self.assertAlmostEqual(word_error_rate("a b", "a b c"), 0.5)
        self.assertAlmostEqual(word_error_rate("a b c", "a b"), 1.0 / 3.0)

    def test_normalization_removes_case_and_punctuation(self):
        self.assertEqual(normalize_text("My name is Sara.", "en"), "my name is sara")
        self.assertEqual(normalize_text("わたしの名前は、サラです。", "ja"), "わたしの名前はサラです")
        self.assertAlmostEqual(word_error_rate("My name is Sara.", "my NAME is sara"), 0.0)

    def test_cer_counts_characters(self):
        self.assertAlmostEqual(
            character_error_rate("わたしの名前はサラです", "わたしの名前はサラでず"),
            1.0 / len("わたしの名前はサラです"))

    def test_keyword_matching_rules(self):
        self.assertFalse(is_correct("sara", "Sarah", "en"))
        self.assertTrue(is_correct("Sara", "My name is Sara.", "en"))
        self.assertTrue(is_correct("ソフィー", "わたしの名前はソフィーです", "ja"))
        self.assertFalse(is_correct("ソフィー", "わたしの名前はサラです", "ja"))


class WhisperTest(unittest.TestCase):
    def test_arguments_from_the_real_yaml(self):
        arguments = whisper_args_from_yaml(WHISPER_YAML, "ja", "prompt")
        expected = [
            "-m", "ggml-large-v3-turbo-q5_0.bin",
            "-bs", "5", "-bo", "5", "-tp", "0", "-tpi", "0.2",
            "-et", "2.4", "-lpt", "-1.0", "-nth", "0.6",
            "-mc", "16384", "-l", "ja", "--prompt", "prompt",
        ]
        self.assertEqual(arguments, expected)
        self.assertEqual(arguments[arguments.index("-mc") + 1], "16384")
        self.assertNotEqual(arguments[arguments.index("-mc") + 1], "0")

    def test_n_max_text_ctx_comes_from_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "whisper.yaml"
            path.write_text(
                "/whisper/whisper_node:\n"
                "  ros__parameters:\n"
                "    model:\n"
                "      filename: test.bin\n"
                "    transcription:\n"
                "      n_max_text_ctx: 512\n",
                encoding="utf-8")
            arguments = whisper_args_from_yaml(path, "en", "", strict=True)
            self.assertEqual(arguments[arguments.index("-mc") + 1], "512")

    def test_arguments_fall_back_when_yaml_is_missing(self):
        arguments = whisper_args_from_yaml(REPO_ROOT / "missing.yaml", "en", "")
        self.assertEqual(arguments[1], "ggml-large-v3-turbo-q5_0.bin")
        self.assertIn("-l", arguments)
        self.assertEqual(arguments[arguments.index("-l") + 1], "en")
        self.assertNotIn("--prompt", arguments)

    def test_default_prompts_match_task_manager(self):
        expected_name = (
            "The guest says their name. Possible names: "
            + ", ".join(name.capitalize() for name in FALLBACK_NAME_LIST) + ".")
        expected_drink = (
            "The guest says their favorite drink. Possible drinks: "
            + ", ".join(FALLBACK_DRINK_LIST) + ".")
        self.assertEqual(build_name_prompt(FALLBACK_NAME_LIST), expected_name)
        self.assertEqual(build_drink_prompt(FALLBACK_DRINK_LIST), expected_drink)
        name_prompt, drink_prompt = english_prompts(REPO_ROOT / "missing.yaml")
        self.assertEqual((name_prompt, drink_prompt), (expected_name, expected_drink))

    def test_parse_json_skips_special_tokens_and_averages_log_probability(self):
        payload = {
            "transcription": [
                {"text": " My name", "tokens": [
                    {"text": "[_BEG_]", "p": 0.1},
                    {"text": " My", "p": 0.5},
                ]},
                {"text": " is Sophie.", "tokens": [
                    {"text": " is", "p": 0.25},
                    {"text": " Sophie", "p": 0.5},
                ]},
            ]
        }
        text, mean_logprob, count = parse_whisper_json(payload)
        self.assertEqual(text, "My name is Sophie.")
        self.assertEqual(count, 3)
        self.assertAlmostEqual(mean_logprob, (math.log(0.5) + math.log(0.25) + math.log(0.5)) / 3.0)


class AnalyzeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.stimuli = build_stimuli(REPO_ROOT / "does-not-exist.yaml")[:2]
        for row in self.stimuli:
            write_sine(self.root / f"{row['utt_id']}_a.wav", seconds=0.3)
        self.playlist_wav = self.root / "playlist_en.wav"
        self.playlist_csv = self.root / "playlist_en.csv"
        self.playlist_rows = build_playlist(
            self.stimuli, self.root, ["a"], "en", seed=0,
            output_wav=self.playlist_wav, output_csv=self.playlist_csv)

    def make_recording(self, offset=1.234, noise_amplitude=300.0, seed=7):
        playlist = read_wav(self.playlist_wav)
        noise = random_noise(len(playlist), noise_amplitude, seed=seed)
        body = [max(-32768, min(32767, int(round(value + disturbance))))
                for value, disturbance in zip(playlist, noise)]
        prefix = random_noise(int(round(offset * SAMPLE_RATE)), noise_amplitude, seed=seed + 1)
        body = [int(round(value)) for value in prefix] + body
        path = self.root / "recording.wav"
        write_wav(path, body)
        return path

    def test_end_to_end_without_asr(self):
        recording = self.make_recording()
        output = self.root / "trials.csv"
        rows, offset = analyze_recording(
            recording, self.playlist_rows, 30, 1.0, "N1", "en",
            offset_s=None, no_asr=True)
        self.assertLessEqual(abs(offset - 1.234), 0.02)
        self.assertEqual(len(rows), 2)
        write_trials_csv(rows, output)
        with output.open(encoding="utf-8") as stream:
            written = list(csv.DictReader(stream))
        self.assertEqual(list(written[0].keys()), TRIAL_COLUMNS)
        amplitude = 0.25 * 32767.0
        noise_rms = 300.0 / math.sqrt(3.0)
        expected = 10.0 * math.log10((amplitude * amplitude / 2.0) / (noise_rms * noise_rms))
        for row in written:
            self.assertEqual(row["snr_valid"], "true")
            self.assertAlmostEqual(float(row["snr_db"]), expected, delta=0.5)
            self.assertEqual(row["correct"], "")
            self.assertEqual(row["hyp"], "")
            self.assertEqual(row["prompt"], "")

    def test_fake_whisper_cli_fills_hypothesis_and_confidence(self):
        recording = self.make_recording(offset=0.0)
        script = self.root / "fake_whisper.py"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "args = sys.argv[1:]\n"
            "prefix = args[args.index('-of') + 1]\n"
            "payload = {'transcription': [{'text': ' My name is Sophie.', 'tokens': [\n"
            "  {'text': '[_BEG_]', 'p': 0.1},\n"
            "  {'text': ' My', 'p': 0.9},\n"
            "  {'text': ' name', 'p': 0.8},\n"
            "  {'text': ' is', 'p': 0.9},\n"
            "  {'text': ' Sophie', 'p': 0.8}]}]}\n"
            "open(prefix + '.json', 'w').write(json.dumps(payload))\n",
            encoding="utf-8")
        script.chmod(0o755)
        rows, _ = analyze_recording(
            recording, [self.playlist_rows[0]], 0, 1.0, "N0", "en",
            offset_s=0.0, whisper_cli=str(script), no_asr=False)
        row = rows[0]
        self.assertEqual(row["hyp"], "My name is Sophie.")
        self.assertEqual(row["correct"], "true")
        self.assertEqual(row["n_tokens"], "4")
        expected = (math.log(0.9) + math.log(0.8) + math.log(0.9) + math.log(0.8)) / 4.0
        self.assertAlmostEqual(float(row["mean_logprob"]), expected, places=5)
        name_prompt, _ = english_prompts()
        self.assertEqual(row["prompt"], name_prompt)

    def test_missing_config_is_an_error_unless_no_asr(self):
        recording = self.make_recording(offset=0.0)
        missing = self.root / "missing.yaml"
        common = (recording, [self.playlist_rows[0]], 0, 1.0, "N0", "en")
        with self.assertRaises(RuntimeError):
            analyze_recording(*common, offset_s=0.0, no_asr=False,
                              hri_params=missing)
        with self.assertRaises(RuntimeError):
            analyze_recording(*common, offset_s=0.0, no_asr=False,
                              whisper_yaml=missing)
        rows, _ = analyze_recording(
            *common, offset_s=0.0, no_asr=True,
            hri_params=missing, whisper_yaml=missing)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt"], "")

    def test_append_writes_header_only_once(self):
        output = self.root / "trials.csv"
        row = trial_row("en", 1.0, "N1", 0, True, -0.1)
        write_trials_csv([row], output)
        write_trials_csv([row], output, append=True)
        lines = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0].split(",")[0], "trial_idx")
        self.assertEqual(len(lines), 3)

    def test_append_rejects_a_different_header(self):
        output = self.root / "old.csv"
        output.write_text("foo,bar\n1,2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            write_trials_csv([trial_row("en", 1.0, "N1", 0, True, -0.1)],
                             output, append=True)

    def test_noise_guard_ignores_the_last_100_ms_before_speech(self):
        playlist = read_wav(self.playlist_wav)
        noise = random_noise(len(playlist), 300.0, seed=11)
        samples = [int(round(value + disturbance))
                   for value, disturbance in zip(playlist, noise)]
        row = self.playlist_rows[0]
        start = int(round(float(row["noise_start_s"]) * SAMPLE_RATE))
        end = int(round(float(row["noise_end_s"]) * SAMPLE_RATE))
        guard = int(round(0.1 * SAMPLE_RATE))
        loud = tone(1000.0, guard / SAMPLE_RATE, amplitude=0.8)
        samples[end - guard:end] = [int(round(value)) for value in loud]
        recording = self.root / "guard.wav"
        write_wav(recording, samples)
        rows, _ = analyze_recording(
            recording, [row], 0, 1.0, "N0", "en", offset_s=0.0, no_asr=True)
        self.assertGreater(float(rows[0]["snr_db"]), 25.0)


class SummaryTest(unittest.TestCase):
    def test_theta_turn_uses_first_ten_point_drop(self):
        rows = []
        for theta, success_count in ((0, 95), (30, 93), (60, 80)):
            for index in range(100):
                rows.append(trial_row("en", 1.0, "N1", theta, index < success_count, -0.1))
        turns = compute_theta_turn(rows, drop_pt=10)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["theta_turn_deg"], 60)

    def test_minus_ninety_drops_if_either_side_drops(self):
        rows = []
        for theta, success_count in ((0, 100), (90, 100), (-90, 50)):
            for index in range(100):
                rows.append(trial_row("en", 1.0, "N1", theta, index < success_count, -0.1))
        turns = compute_theta_turn(rows, drop_pt=10)
        self.assertEqual(turns[0]["theta_turn_deg"], 90)

    def test_summary_counts_and_accuracy(self):
        rows = [trial_row("en", 1.0, "N1", 0, index < 9, -0.1) for index in range(10)]
        summary = compute_summary(rows)
        self.assertEqual(summary[0]["n"], 10)
        self.assertAlmostEqual(summary[0]["accuracy_pct"], 90.0)

    def test_wilson_interval_known_value(self):
        low, high = wilson_interval(95, 100)
        self.assertAlmostEqual(low, 0.8882, places=3)
        self.assertAlmostEqual(high, 0.9785, places=3)

    def test_invalid_snr_is_ranked_as_minus_infinity(self):
        rows = [
            trial_row("en", 1.0, "N2", 180, True, -0.1, snr=10.0),
            trial_row("en", 1.0, "N2", 180, True, -0.1, snr=20.0),
            trial_row("en", 1.0, "N2", 180, True, -0.1, snr=30.0),
            trial_row("en", 1.0, "N2", 180, True, -0.1, snr=-99.0,
                      snr_valid=False),
        ]
        summary = compute_summary(rows)[0]
        self.assertEqual(summary["n_snr_invalid"], 1)
        self.assertAlmostEqual(summary["snr_median"], 15.0)

    def test_invalid_snr_quantile_is_blank_in_csv(self):
        rows = [
            trial_row("en", 1.0, "N2", 180, True, -0.1, snr=10.0),
            trial_row("en", 1.0, "N2", 180, True, -0.1, snr=-99.0,
                      snr_valid=False),
            trial_row("en", 1.0, "N2", 180, True, -0.1, snr=-99.0,
                      snr_valid=False),
            trial_row("en", 1.0, "N2", 180, True, -0.1, snr=-99.0,
                      snr_valid=False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            write_summary_csv(compute_summary(rows), path)
            with path.open(encoding="utf-8") as stream:
                written = list(csv.DictReader(stream))[0]
        self.assertEqual(written["n_snr_invalid"], "3")
        self.assertEqual(written["snr_median"], "")

    def test_roc_auc_known_values(self):
        points, auc = roc_curve([0.9, 0.8, 0.4, 0.3], [True, True, False, False])
        self.assertAlmostEqual(auc, 1.0)
        self.assertTrue(points)
        _, auc = roc_curve([0.9, 0.8, 0.4, 0.3], [True, False, True, False])
        self.assertAlmostEqual(auc, 0.75)
        _, auc = roc_curve([0.9, 0.8], [True, True])
        self.assertIsNone(auc)

    def test_roc_keeps_minus_infinity_and_reports_excluded(self):
        points, auc, excluded = roc_curve_detail(
            [0.9, "", "-inf"], [True, True, False])
        self.assertEqual(excluded, 1)
        self.assertAlmostEqual(auc, 1.0)
        self.assertIn(float("-inf"), [point[0] for point in points])

    def test_roc_groups_include_languages(self):
        rows = [
            trial_row("en", 1.0, "N1", 0, True, -0.1),
            trial_row("en", 1.0, "N1", 0, False, -0.2),
            trial_row("ja", 1.0, "N1", 0, True, "-inf"),
            trial_row("ja", 1.0, "N1", 0, False, ""),
        ]
        groups = roc_groups(rows)
        self.assertEqual([group[0] for group in groups], ["all", "en", "ja"])
        self.assertEqual(groups[0][3], 1)  # all の空スコア
        self.assertEqual(groups[2][3], 1)  # ja の空スコア

    def test_summary_cli_writes_files(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trials = directory / "trials.csv"
            rows = []
            for theta, success_count in ((0, 95), (30, 93), (60, 80)):
                for index in range(100):
                    rows.append(trial_row("en", 1.0, "N1", theta, index < success_count, -0.1))
            write_trials_csv(rows, trials)
            result = main(["summary", str(trials), "--no-plot", "--out-dir", str(directory)])
            self.assertEqual(result, 0)
            self.assertTrue((directory / "summary.csv").exists())
            self.assertTrue((directory / "theta_turn.csv").exists())
            self.assertTrue((directory / "roc.csv").exists())
            with (directory / "theta_turn.csv").open(encoding="utf-8") as stream:
                turn_rows = list(csv.DictReader(stream))
            self.assertEqual(turn_rows[0]["theta_turn_deg"], "60")
            with (directory / "roc.csv").open(encoding="utf-8") as stream:
                roc_rows = list(csv.DictReader(stream))
            self.assertIn("excluded", roc_rows[0])
            self.assertEqual({row["group"] for row in roc_rows}, {"all", "en"})


class DirectivityTest(unittest.TestCase):
    def test_theoretical_values(self):
        self.assertAlmostEqual(supercardioid_db(90), -8.64, delta=0.1)
        self.assertAlmostEqual(supercardioid_db(180), -11.70, delta=0.1)
        self.assertAlmostEqual(supercardioid_db(0), 0.0, delta=0.01)

    def test_band_list(self):
        self.assertEqual(THIRD_OCTAVE_CENTERS[0], 125)
        self.assertEqual(THIRD_OCTAVE_CENTERS[-1], 6300)
        self.assertEqual(len(THIRD_OCTAVE_CENTERS), 18)

    @unittest.skipUnless(HAVE_NUMPY, "numpy が無い")
    def test_theory_is_relative_to_the_reference_angle(self):
        recordings = {
            0.0: tone(1000.0, 1.0, amplitude=1.0),
            90.0: tone(1000.0, 1.0, amplitude=0.37),
            180.0: tone(1000.0, 1.0, amplitude=0.26),
        }
        rows = compute_directivity(
            recordings, start_s=0.0, duration_s=1.0, reference_deg=90.0)
        broadband = {row["theta_deg"]: row for row in rows
                     if row["band_hz"] == "broadband"}
        self.assertAlmostEqual(broadband[90.0]["theory_db"], 0.0, delta=0.01)
        self.assertAlmostEqual(broadband[0.0]["theory_db"], 8.64, delta=0.05)
        self.assertAlmostEqual(broadband[0.0]["measured_db"], 8.64, delta=0.05)

    @unittest.skipUnless(HAVE_NUMPY, "numpy が無い")
    def test_measured_broadband_matches_theory(self):
        recordings = {
            0.0: tone(1000.0, 1.0, amplitude=1.0),
            90.0: tone(1000.0, 1.0, amplitude=0.37),
            180.0: tone(1000.0, 1.0, amplitude=0.26),
        }
        rows = compute_directivity(recordings, start_s=0.0, duration_s=1.0)
        broadband = {row["theta_deg"]: row for row in rows if row["band_hz"] == "broadband"}
        self.assertAlmostEqual(broadband[90.0]["measured_db"], -8.64, delta=0.05)
        self.assertAlmostEqual(broadband[180.0]["measured_db"], -11.70, delta=0.05)
        self.assertAlmostEqual(broadband[90.0]["diff_db"], 0.0, delta=0.05)


class NoiseTest(unittest.TestCase):
    @unittest.skipUnless(HAVE_NUMPY, "numpy が無い")
    def test_babble_has_requested_length_and_speakers(self):
        from robot_audio_bench.noise import collect_speakers, make_babble
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(8):
                speaker = root / f"spk{index}"
                speaker.mkdir()
                write_sine(speaker / f"spk{index}-0001.wav", frequency=200.0 + index * 50.0,
                           seconds=0.2, amplitude=0.2)
            groups = collect_speakers(root)
            self.assertEqual(len(groups), 8)
            output = root / "babble.wav"
            info = make_babble(root, speakers=8, duration=0.5, seed=1, output=output)
            self.assertEqual(info["speaker_count"], 8)
            frames, rate, channels, width = wav_info(output)
            self.assertEqual((frames, rate, channels, width), (8000, SAMPLE_RATE, 1, 2))

    @unittest.skipUnless(HAVE_NUMPY, "numpy が無い")
    def test_pink_noise_has_requested_length(self):
        from robot_audio_bench.noise import make_pink
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pink.wav"
            make_pink(duration=0.5, seed=1, output=output)
            frames, rate, channels, width = wav_info(output)
            self.assertEqual((frames, rate, channels, width), (8000, SAMPLE_RATE, 1, 2))

    def test_babble_without_numpy_reports_it(self):
        if HAVE_NUMPY:
            self.skipTest("numpy がある")
        from robot_audio_bench.noise import make_babble
        with self.assertRaises(ImportError):
            make_babble(".", duration=0.1)


if __name__ == "__main__":
    unittest.main()
