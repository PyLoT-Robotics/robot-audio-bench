"""Exercise WAV -> playlist -> recording -> SNR CSV without ROS or ASR.

Tone bursts stand in for speech. Results test the processing path, not ASR quality.
"""
import argparse
from pathlib import Path
import random
from robot_audio_bench.stimuli import build_stimuli,write_stimuli_csv
from robot_audio_bench.playlist import build_playlist
from robot_audio_bench.wav import tone,write_wav,read_wav
from robot_audio_bench.analyze import analyze_recording,write_trials_csv
from robot_audio_bench.summary import compute_summary,write_summary_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/demo'))
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    wav_dir=out/'wav';wav_dir.mkdir(exist_ok=True)
    stimuli=[s for s in build_stimuli() if s['lang']=='en'][:2]
    write_stimuli_csv(stimuli,out/'stimuli.csv')
    for row in stimuli:
        write_wav(wav_dir/(row['utt_id']+'_demo.wav'),tone(330.,.5,amplitude=.2))
    rows=build_playlist(stimuli,wav_dir,['demo'],'en',seed=0,
                        output_wav=out/'playlist.wav',output_csv=out/'playlist.csv')
    recording=read_wav(out/'playlist.wav')
    rng=random.Random(0)
    for i,sample in enumerate(recording):
        recording[i]=max(-32768,min(32767,sample+rng.randint(-200,200)))
    write_wav(out/'recording.wav',recording)
    trials,offset=analyze_recording(out/'recording.wav',rows,theta_deg=0.,distance_m=1.,
                                    noise='N0',lang='en',no_asr=True)
    write_trials_csv(trials,out/'trials.csv')
    write_summary_csv(compute_summary(trials),out/'summary.csv')
    assert len(trials)==2 and all(row['snr_valid']=='true' for row in trials)
    print(f'{len(trials)} synthetic trials; synchronization offset={offset:.3f}s; results in {out}')


if __name__=='__main__':main()
