"""Shared figure style. Importing this sets Arial as the default font.

Arial comes from the conda env (mscorefonts in environment.yml), so the figures look the same
on machines without a system Arial (e.g. NERSC). matplotlib doesn't search the env's fonts/
folder, so the files are registered here.
"""

import sys
import glob
import matplotlib as mpl
import matplotlib.font_manager as fm

for f in glob.glob(f'{sys.prefix}/fonts/arial*.ttf'):
    fm.fontManager.addfont(f)
mpl.rcParams['font.family'] = 'Arial'
