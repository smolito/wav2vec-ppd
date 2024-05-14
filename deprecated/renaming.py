import os

path2folder = r"C:\Users\smold\PycharmProjects\w2v_intelligibility\PD_intelligibilityData\15 Young Healthy Control\Alessandro M\pr_split"

print(os.listdir(path2folder))

lst = os.listdir(path2folder)

# You can remove the first character from each string in the list like this:
new_lst = [s[1:] for s in lst]

nms = []
for s in new_lst:
    s = str.replace(s, ".wav", "")
    nms.append(s + "-untitled.wav")

print(nms)

for file, nm in zip(lst, nms):
    old_nm = os.path.join(path2folder, file)
    new_nm = os.path.join(path2folder, nm)

    os.rename(old_nm, new_nm)
