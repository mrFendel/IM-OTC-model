from glob import glob
import zipfile as zf
from pprint import pprint
from joblib import Parallel, delayed

pprint(glob('data/**/*.zip', recursive=True))


def unpack(path):
    with zf.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(path[:-4])
    return None


res = Parallel(n_jobs=-1, verbose=10)(delayed(unpack)(x) for x in glob('data/**/*.zip', recursive=True))
