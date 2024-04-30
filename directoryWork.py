import fnmatch
import os
import jiwer
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("_transcripts/pr_split.csv")
    # dont include nan values from transcript
    df = df.dropna(subset=["transcript"])
    # wer, cer

    cer = []
    new_wer = []
    for reference, hypothesis in zip(df.word, df.transcript):
        print(reference, hypothesis)

        cer.append(jiwer.cer(reference, hypothesis))
        new_wer.append(jiwer.wer(reference, hypothesis))

    df = df.assign(cer=cer, wer=new_wer)
    print(df.head())
    df.to_csv("_transcripts/pr_split_clean.csv")
