from tqdm import tqdm
from glob import glob
import os


def clean(path):
    with open(path, "r+") as f:
        rows = f.readlines()
        f.seek(0)
        f.truncate()
        f.writelines(rows[2:])


dirs = glob('data/**/*.csv', recursive=True)
for dir in dirs:
    if os.path.isdir(dir):
        os.rename(dir, dir[:-4])

paths = glob('data/**/*.csv', recursive=True)

for p in tqdm(paths[102:]):
    if p[-10:] == 'trades.csv':
        clean(p)
    else:
        pass
