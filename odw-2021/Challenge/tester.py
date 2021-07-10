import os
import shutil
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.frame import read_frame
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from colorama import Fore, Back, Style
from pycbc.waveform import get_fd_waveform
from pycbc.filter import matched_filter
from pycbc.vetoes import power_chisq
from pycbc.events.ranking import newsnr
import matplotlib.pyplot as plt
import pandas as pd

my_path = os.path.dirname(os.path.abspath(__file__))

class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    LIGHTGREEN = "\033[38;5;40m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
    ITALICS = "\033[3m"
    ITAB = "\033[3;1m"
    BLUEB = "\033[34;1m"
    GREENB = "\033[32;1m"
    CYANB = "\033[36;1m"
    REDB = "\033[91;1m"

# stdoutOrigin = sys.stdout
# sys.stdout = open("Challenge4_output.txt", "w")

# The real stuff of detecting signals
masses = []
dth = 8  # BBH merger signal threshold
BBH_dg = 750
# Assuming that for any apparent signal detected there will be no other merger
# peak within .36secs on either side of the said tentative detection as\
# it is implausible

nbins = 26  # for the reduced chisq filtering of removing glitches
dof = nbins * 2 - 2  # for the reduced chisq filtering of removing glitches

for x in range(10, 15):
    #os.mkdir("{}".format(x))
    path = os.path.join(my_path, str(x))
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)           # Removes all the subdirectories!
        os.makedirs(path)
    print(path)
    print(Style.RESET_ALL)
    print(color.BOLD + "Individual masses of the BBHs -- ", x, "solar masses" + color.END)
    print(Style.RESET_ALL)
