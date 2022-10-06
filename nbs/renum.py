#!/usr/bin/env python
"Renumber files starting at 01"

from fastcore.utils import *
from shutil import move

i = 1
for p in sorted(Path().ls()):
    pre = re.findall(r'^(\d+)[a-z]?_(.*)', str(p))
    if not pre: continue
    new = f'{i:02d}_{pre[0][1]}'
    move(p, new)
    i+=1

