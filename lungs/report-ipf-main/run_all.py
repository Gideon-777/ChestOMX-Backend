import subprocess
import sys
from itertools import chain, combinations


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


name = sys.argv[1]

all_flags = ['dicom', 'lobes3d', 'lobes', 'ggo_consolidation', 'summary']

for flags in powerset(all_flags):
    cmd = 'python process.py ' + " ".join(['--' + f for f in flags]) + ' --dir ' + name + \
          ' --out ' + name + '_' + "_".join(flags)

    print(cmd)
    subprocess.Popen(cmd)
