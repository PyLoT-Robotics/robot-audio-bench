from glob import glob
from setuptools import setup, find_packages
setup(name='robot_audio_bench', version='0.1.0', description='Audio benchmarks for robot microphones and speech recognition',
      packages=find_packages(exclude=['test']), python_requires='>=3.10',
      install_requires=['PyYAML>=6'], extras_require={'test':['pytest', 'numpy', 'matplotlib'], 'plots':['matplotlib'], 'flac':['soundfile']},
      license='Apache-2.0', maintainer='PyLoT Robotics', maintainer_email='163533636+PyLoT-Robotics@users.noreply.github.com',
      package_data={'robot_audio_bench': ['config/*.yaml']},
      entry_points={'console_scripts':['robot-audio-bench = robot_audio_bench.cli:main']})
