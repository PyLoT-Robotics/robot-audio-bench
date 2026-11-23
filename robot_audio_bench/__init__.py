"""Hear-and-Face ベンチ（Issue #261）用の音声ツール。

このパッケージのモジュールは import 時に numpy / matplotlib / yaml を読まない。
重い依存は必要な関数の中で遅延 import する（test_no_ros_import.py の制約）。
"""
