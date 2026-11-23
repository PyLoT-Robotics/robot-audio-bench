# Robot Audio Bench

[English](../README.md) · 日本語

ロボットのマイクと音声認識を、話者の角度・距離・雑音条件ごとに評価するPython CLIです。
`pylot_robocup2025` の Hear-and-Face ベンチから独立させたprivateリポジトリです。
ROS、PyLoTのワークスペース、GPU、APIキーは基本機能に不要です。

## インストールとクイックスタート

Python 3.10+。以下はリポジトリのルートで実行します。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
robot-audio-bench stimuli --output stimuli.csv
python examples/synthetic_demo.py --output results/demo
python -m pytest -q
```

デモは合成トーンを発話の代わりに使い、WAV → 再生リスト → 録音解析 →
SNRと集計CSVの経路を確認します。合成音による認識精度・実機性能の主張はしません。
CLIは `python -m robot_audio_bench` でも実行できます。

## 実録音の測定

1. `stimuli` で生成したCSVの各発話を人に録音してもらいます。
   `wav` 列の `<speaker>` を話者名（例: alice）に置き換えたファイル名で保存します。
2. 16 kHz / mono / signed int16 のWAVを用意し、再生リストを作ります。
3. 音源を各角度・距離に配置し、実際のロボットマイクで録音します。
4. 同期トーンで録音と再生リストを合わせ、雑音区間と発話区間を比較します。

```bash
robot-audio-bench playlist stimuli.csv --wav-dir recordings --speakers alice --lang en --out-dir playlist
robot-audio-bench analyze --recording rec.wav --playlist playlist/playlist_en.csv \
  --theta-deg 60 --distance-m 1 --noise N0 --lang en --no-asr --out trials.csv
robot-audio-bench summary trials.csv --out-dir summary --no-plot
```

`playlist` の先頭には1 kHz・0.5秒の同期トーンと1秒の無音があります。
各試行の雑音区間は再生リスト上では無音です。雑音は現場で別途再生し、録音側に
含めてください。`--offset-s` で同期位置を明示することもできます。
SNRが算出不能な試行は `snr_valid=false` となり、成功した測定として扱いません。
実環境の音圧レベルは別途校正してください。N0/N1/N2は条件ラベルです。

## 音声認識と追加機能

実音声認識には自分でビルドした **whisper.cppのwhisper-cli** とモデルが必要です。
元実装の対象はwhisper.cpp v1.9.2のJSON full出力です。本リポジトリはモデルを取得しません。
`analyze` の `--no-asr` を外して指定します。

```bash
robot-audio-bench analyze --recording rec.wav --playlist playlist/playlist_en.csv \
  --theta-deg 60 --distance-m 1 --noise N0 --lang en \
  --whisper-cli /path/to/whisper-cli --model /path/to/model.bin --out trials.csv
```

`robot_audio_bench/config/dialogue.yaml` と `whisper.yaml` がインストールに同梱され、
作業ディレクトリに依存しません。`--dialogue-config`（互換名: `--hri-params`）と
`--whisper-yaml` で差し替えられます。設定のROS風ラッパーは互換性のため保持しています。
初版の刺激生成・プロンプト分類は名前／飲み物の2カテゴリです。別の文を評価する場合は
刺激CSV・参照テキストと、この2カテゴリのプロンプト設定を自分で用意してください。

- `noise pink` / `noise babble`: ピンク雑音、手元のLibriSpeechからバブル雑音を生成。
- `directivity`: 1/3オクターブ帯域別の角度特性。NumPyを使用。
- `summary`: 条件別SNR、正解率とWilson区間、WER/CER、ROC、正解率低下角度。
- 図の生成: `pip install '.[plots]'`。FLAC読み取り: `pip install '.[flac]'`。

認識テストはパーサ・引数構築・合成データを検証します。実モデル推論や実マイク性能は
今回の検証に含みません。ASRキャッシュを使う場合はモデル変更時に別ディレクトリを
指定してください。同じパスのモデル内容が変わったことはキャッシュキーに反映されません。

## ライセンスと出典

Apache-2.0。元コードの所在とコミットは [PROVENANCE.md](../PROVENANCE.md)。
外部モデル、音声コーパス、個人の録音は含めていません。
