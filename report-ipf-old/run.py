import subprocess
from pathlib import Path
import glob2
import shutil
import os
from distutils.dir_util import copy_tree

#if os.path.isdir('./input'):
#    shutil.rmtree('./input')

#os.mkdir('./input')

#if not os.path.isdir('/workspace/output/reports'):
#    os.mkdir('/workspace/output/reports')

#copy_tree('/workspace/output', './input')
#copy_tree('/workspace/input', './input')

for file in glob2.glob('./input/*'):
    # name = Path(file).stem[:-4]
    name = Path(file).stem#[:-4]
    print(name, 'name')
    # cmd = "python process.py --header_name " + name + " --lungs --lobes3d --lobes --infiltration --summary --exam " + name + " --out " + name
    # cmd = "python process.py --header_name " + name + " --lungs --lobes3d --lobes --ipf --summary --exam " + name + " --out " + name
    cmd = "python process.py --header_name " + name + " --lungs --lobes3d --lobes --summary --exam " + name + " --out " + name
    print(cmd)
    proc = subprocess.Popen(cmd.split(' '))
    proc.communicate()

#    shutil.move('./' + name + '.html', '/workspace/output/reports/' + name + '.html')

#shutil.rmtree('./input')
