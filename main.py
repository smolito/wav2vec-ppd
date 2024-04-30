import os
import time
import IPython
from IPython.display import Audio
import csv
import numpy
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio
import fnmatch
import jiwer
from pydub import AudioSegment
from pydub.silence import split_on_silence
import wave

matplotlib.rcParams["figure.figsize"] = [16.0, 5.5]
matplotlib.rcParams["figure.dpi"] = 200.0
matplotlib.rcParams["image.interpolation"] = "none"

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torchaudio.list_audio_backends())
print(torch.__version__)
print(torchaudio.__version__)
print("torchaudio backend:", torchaudio.get_audio_backend())
print(device)

# choose a control group to analyze
folder2do = r"28 People with Parkinson's disease"
FOLDERS_ROOT = r"PD_intelligibilityData/" + folder2do

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
# print("Sample Rate:", bundle.sample_rate)
# print("Labels:", bundle.get_labels())

model = bundle.get_model().to(device)
# print(model.__class__)

with open('_assets/reference_b.txt', 'r') as file:
    reference_b = file.read().replace('\n', '')
    file.close()

with open('_assets/reference_pr.txt', 'r') as file:
    reference_pr = file.read().replace('\n', '')
    file.close()

with open('_assets/reference_fb.txt', 'r') as file:
    reference_fb = file.read().replace('\n', '')
    file.close()


def rows2csv(data2write):
    if not os.path.exists(csvFILE):
        os.makedirs("_transcripts", exist_ok=True)
    with open(csvFILE, "w", newline="", encoding='UTF8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data2write)
        csvfile.close()


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]

        return "".join([self.labels[i] for i in indices])


if __name__ == "__main__":

    data2csv = []
    start_time = time.time()
    for folder in os.listdir(os.path.join(FOLDERS_ROOT)):
        folder_time = time.time()
        print("working on: " + folder)
        for file in os.listdir(os.path.join(FOLDERS_ROOT, folder)):

            one_file_time_start = time.time()

            if fnmatch.fnmatch(file, 'B*.wav'):
                recordingType = "B"
            elif fnmatch.fnmatch(file, 'PR*.wav'):
                recordingType = "PR"
            elif fnmatch.fnmatch(file, 'D*.wav'):
                recordingType = "D"
            elif fnmatch.fnmatch(file, 'FB*.wav'):
                recordingType = "FB"
            elif fnmatch.fnmatch(file, 'VA*.wav'):
                recordingType = "VA"
            elif fnmatch.fnmatch(file, 'VE*.wav'):
                recordingType = "VE"
            elif fnmatch.fnmatch(file, 'VI*.wav'):
                recordingType = "VI"
            elif fnmatch.fnmatch(file, 'VO*.wav'):
                recordingType = "VO"
            elif fnmatch.fnmatch(file, 'VU*.wav'):
                recordingType = "VU"
            else:
                recordingType = "english"  # for testing

            if fnmatch.fnmatch(file, '*.wav'):
                print("... listening to: " + file)
                waveform, sample_rate = torchaudio.load(os.path.join(FOLDERS_ROOT, folder, file))
                waveform = waveform.to(device)

                if sample_rate != bundle.sample_rate:
                    waveform = torchaudio.functional.resample(waveform,
                                                              sample_rate,
                                                              bundle.sample_rate)

                with torch.inference_mode():
                    features, _ = model.extract_features(waveform)

                    # LAYERS VISUALIZATION
                    if not os.path.exists("_plots/" + FOLDERS_ROOT + "/" + folder + "/" + recordingType):
                        os.makedirs("_plots/" + FOLDERS_ROOT + "/" + folder + "/" + recordingType)

                    # fig, ax = plt.subplots(len(features), 1, figsize=(40, 10 * len(features)))
                    # print("delka ax: " + str(len(ax)))
                    # for i, feats in enumerate(features):
                    #     print("print feats size/shape")
                    #     get_tensor_shape = feats.shape
                    #     get_tensor_shape = list(get_tensor_shape)
                    #     print(get_tensor_shape)
                    #     print("print single feats i: " + str(i))
                    #     # print(feats[0])
                    #
                    #     plt.imshow(feats[0].cpu(),
                    #                vmin=torch.min(feats[0].cpu()),
                    #                vmax=torch.max(feats[0].cpu()),
                    #                cmap="ocean",  # jet, turbo, rainbow, cubehelix
                    #                aspect="auto")
                    #     plt.colorbar()
                    #     # plt.tight_layout()
                    #     plt.title(f"Příznaková vrstva transformeru {i + 1}")
                    #     plt.xlabel("Dimenze příznaku")
                    #     plt.ylabel("rámec - čas")
                    #
                    #     # path2fig = "_plots/" + FOLDERS_ROOT + "/"+folder + "/" + recordingType "/" + file + "_imshow" +
                    #     # "_layer_" + str(i) + ".png"
                    #     path2fig = os.path.join("_plots", FOLDERS_ROOT, folder, recordingType) + "/" + file + "_imshow" + "_layer_" + str(i) + ".png"
                    #     plt.savefig(path2fig)
                    #     plt.close("all")

                # FEATURE CLASSIFICATION (in logits, not probability)
                print("... doing classification")
                with torch.inference_mode():
                    inference, _ = model(waveform)

                    # # FEATURE CLASSIFICATION VIS
                    # plt.imshow(inference[0].cpu().T)
                    # plt.title("Výsledek klasifikace")
                    # plt.xlabel("Čas")
                    # plt.ylabel("Třída")
                    # plt.colorbar(orientation="horizontal")
                    # # plt.tight_layout()
                    #
                    # path2fig = os.path.join("_plots", FOLDERS_ROOT, folder, recordingType) + "/" + file + "_classification" + ".png"
                    # plt.savefig(path2fig)
                    # plt.close("all")
                    # print("Class labels:", bundle.get_labels())

                # GENERATING TRANSCRIPTS
                print("... generating transcript")
                decoder = GreedyCTCDecoder(labels=bundle.get_labels())
                transcript = decoder(inference[0])

                # print("transcript length: ", transcript.__len__())
                # print("transcript: ", transcript)

                # PREP DATA FOR CSV
                word_error_rate = 0
                character_error_rate = 0
                get_unique_words = transcript.split('|')
                get_unique_words = (" ".join(get_unique_words)).lower()

                if recordingType == "PR":
                    word_error_rate = jiwer.wer(reference_pr, get_unique_words)
                    character_error_rate = jiwer.cer(reference_pr, get_unique_words)
                elif recordingType == "B":
                    word_error_rate = jiwer.wer(reference_b, get_unique_words)
                    character_error_rate = jiwer.cer(reference_b, get_unique_words)
                elif recordingType == "FB":
                    word_error_rate = jiwer.wer(reference_fb, get_unique_words)
                    character_error_rate = jiwer.cer(reference_fb, get_unique_words)
                else:
                    word_error_rate = float("NaN")
                    character_error_rate = float("NaN")

                one_file_time = time.time()
                one_file_time = round(one_file_time - one_file_time_start, 2)

                # if not word_error_rate == float("NaN") and not character_error_rate == float("NaN"):
                transcript_data = [folder2do, folder, recordingType, get_unique_words, word_error_rate, character_error_rate, one_file_time]
                data2csv.append(transcript_data)

                print("file was done in " + str(one_file_time) + "s")

        print("folder finished in " + str(round(time.time() - folder_time, 2)) + "s")

    print("FINISHED IN " + str(round(time.time() - start_time, 2)) + "s")

    header = ["group", "name", "file_code", "transcribed text", "wer", "cer", "t (s)"]
    csvFILE = "_transcripts/" + folder2do + ".csv"

    rows2csv(data2csv)
