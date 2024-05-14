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
from jiwer import wer
import wave

matplotlib.rcParams["figure.figsize"] = [9.0, 4.3]
matplotlib.rcParams["figure.dpi"] = 175.0
matplotlib.rcParams["image.interpolation"] = "none"


def read_pr_reference_file(path2file):
    # prep words reference list
    with open(path2file, "r") as f:
        rawlines = f.readlines()
        f.close()

    refs = []
    for line in rawlines:
        line = str.replace(line, "\n", "")
        line = str.replace(line, " ", "")
        refs.append(line)

    return refs


torch.random.manual_seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torchaudio.list_audio_backends())
print(torch.__version__)
print(torchaudio.__version__)
print("torchaudio backend:", torchaudio.get_audio_backend())
# backend: soundfile for Win, Sox for Linux
print(device)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
print("Sample Rate:", bundle.sample_rate)
print("Popisky (labely): ", bundle.get_labels())

model = bundle.get_model().to(device)
print(model.__class__)


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

    pd_intel_root = r"PD_intelligibilityData/" # run through all the data
    # pd_intel_root = r"quickrun/" # test run
    data2csv = []
    start_time = time.time()

    for root_folder in os.listdir(pd_intel_root):

        print(root_folder)

        for group_folder in os.listdir(os.path.join(pd_intel_root, root_folder)):
            print(group_folder)

            for persons_files in os.listdir(os.path.join(pd_intel_root, root_folder, group_folder)):
                folder_time = time.time()

                if fnmatch.fnmatch(persons_files, 'pr_split*'):
                    # print(persons_files)

                    references_pr_loop = read_pr_reference_file("_assets/split_pr.txt")

                    skipped = [f for f in os.listdir(os.path.join(pd_intel_root,
                                                                  root_folder,
                                                                  group_folder,
                                                                  persons_files))
                               if f == "skipped.txt"]

                    if skipped:
                        path2skipped = os.path.join(pd_intel_root, root_folder, group_folder, persons_files,
                                                    r"skipped.txt")
                        print(path2skipped)
                        with open(path2skipped, "r") as file:
                            skipped_words = file.readlines()
                            file.close()

                        skipped_words_clean = []
                        for line in skipped_words:
                            line = str.replace(line, "\n", "")
                            line = str.replace(line, " ", "")
                            skipped_words_clean.append(line)

                        for word in skipped_words_clean:
                            references_pr_loop.remove(word)
                        print(references_pr_loop)

                    refs_with_id = []
                    for w in enumerate(references_pr_loop):
                        refs_with_id.append(w)

                    print(refs_with_id)

                    # iterate over split words in pr_split folders
                    for word_wav in os.listdir(os.path.join(pd_intel_root,
                                                            root_folder,
                                                            group_folder,
                                                            persons_files)):
                        one_file_time_start = time.time()

                        if fnmatch.fnmatch(word_wav, '*.wav'):
                            print("... listening to: " + root_folder + " " + group_folder + " " + persons_files + " " +
                                  word_wav)
                            waveform, sample_rate = torchaudio.load(os.path.join(pd_intel_root,
                                                                                 root_folder,
                                                                                 group_folder,
                                                                                 persons_files,
                                                                                 word_wav))
                            waveform = waveform.to(device)

                            if sample_rate != bundle.sample_rate:
                                waveform = torchaudio.functional.resample(waveform,
                                                                          sample_rate,
                                                                          bundle.sample_rate)

                            # convert file num to int id corresponding to references list
                            name2id = int(str.replace(word_wav, "-untitled.wav", "")) - 1
                            word_with_id = [w for w in refs_with_id if w[0] == name2id]

                            with torch.inference_mode():
                                features, _ = model.extract_features(waveform)
                                # print("delka features: ", len(features))

                                # LAYERS VISUALIZATION
                                if not os.path.exists(
                                        os.path.join("_plots/pr_split", root_folder, group_folder, (str(word_with_id[0][0]) + "_" + word_with_id[0][1]))):
                                    os.makedirs(
                                        os.path.join("_plots/pr_split", root_folder, group_folder, (str(word_with_id[0][0]) + "_" + word_with_id[0][1])))

                                for i, feats in enumerate(features):
                                    # print("print feats size/shape")
                                    # get_tensor_shape = feats.shape
                                    # get_tensor_shape = list(get_tensor_shape)
                                    # print(get_tensor_shape)
                                    # print("print single feats i: " + str(i))
                                    # print(feats[0])

                                    plt.imshow(feats[0].cpu(),
                                               vmin=torch.min(feats[0].cpu()),
                                               vmax=torch.max(feats[0].cpu()),
                                               cmap="ocean",  # jet, turbo, rainbow, cubehelix
                                               aspect="auto")
                                    plt.colorbar()
                                    # plt.tight_layout()
                                    plt.title(f"Příznaková vrstva transformeru {i + 1}")
                                    plt.xlabel("Dimenze příznaku")
                                    plt.ylabel("rámec - čas")

                                    path2fig = os.path.join("_plots/pr_split", root_folder, group_folder,
                                                            (str(word_with_id[0][0]) + "_" + word_with_id[0][1])) \
                                               + "/" + word_with_id[0][1] + "_imshow" + "_layer_" + str(i) + ".png"
                                    plt.savefig(path2fig)
                                    plt.close("all")

                            # FEATURE CLASSIFICATION (in logits, not probability)
                            print("... doing classification")
                            with torch.inference_mode():
                                inference, _ = model(waveform)
                                print("delka inference: ", len(inference))

                                # FEATURE CLASSIFICATION VIS
                                plt.imshow(inference[0].cpu().T)
                                plt.title("Výsledek klasifikace")
                                plt.xlabel("Čas")
                                plt.ylabel("Třída")
                                plt.colorbar(orientation="vertical")
                                # plt.tight_layout()

                                path2fig = os.path.join("_plots/pr_split", root_folder, group_folder, (str(word_with_id[0][0]) + "_" + word_with_id[0][1])) \
                                           + "/" + word_with_id[0][1] + "_classification" + ".png"
                                plt.savefig(path2fig)
                                plt.close("all")
                                # print("Class labels:", bundle.get_labels())

                            # GENERATING TRANSCRIPTS
                            print("... generating transcript")
                            decoder = GreedyCTCDecoder(labels=bundle.get_labels())
                            transcript = decoder(inference[0])

                            # PREP DATA FOR CSV
                            get_unique_words = transcript.split('|')  # mezera mezi slovy
                            get_unique_words = (" ".join(get_unique_words)).lower()

                            word_error_rate = jiwer.wer(word_with_id[0][1], get_unique_words)
                            character_error_rate = jiwer.cer(word_with_id[0][1], get_unique_words)

                            print("WER: ", word_with_id[0][1], get_unique_words, word_error_rate)
                            print("CER: ", word_with_id[0][1], get_unique_words, character_error_rate)

                            one_file_time = time.time()
                            one_file_time = round(one_file_time - one_file_time_start, 2)

                            transcript_data = [root_folder, group_folder, word_with_id[0][0], word_with_id[0][1],
                                               get_unique_words, word_error_rate, character_error_rate, one_file_time]
                            data2csv.append(transcript_data)

                    print("file was done in " + str(one_file_time) + "s")

            print("folder finished in " + str(round(time.time() - folder_time, 2)) + "s")

        print("FINISHED IN " + str(round(time.time() - start_time, 2)) + "s")

    header = ["group", "name", "word_id", "word", "transcript", "wer", "cer", "t"]
    csvFILE = "_transcripts/" + "pr_split" + ".csv"

    rows2csv(data2csv)
