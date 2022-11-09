import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import signal
from scipy import stats
import mne 

def plot_TF(raw, ONSET=540, OFFSET=630, SZ_ON=589, CH_IDX=0):
    idx_ = np.logical_and(raw.times > ONSET, raw.times < OFFSET)
    raw.times[idx_]
    dat = raw.get_data()[:, idx_]

    f, t, Zxx = signal.stft(
    dat[CH_IDX, :], fs=250, nperseg=128, noverlap=125
    )
    z_abs = np.abs(Zxx)

    plt.figure(figsize=(6,6), dpi=300)
    plt.subplot(212)
    plt.plot(raw.times[idx_] - raw.times[idx_][0], dat[CH_IDX, :], linewidth=0.1)
    plt.xlabel("Time [s]")
    plt.xlim(0, 90)
    plt.subplot(211)
    plt.imshow(stats.zscore(z_abs), aspect='auto')
    plt.gca().invert_yaxis()
    plt.title("Time Frequency Plot")
    plt.axvline(x=np.where(t>(SZ_ON-ONSET))[0][0], label="Seizure Onset", color="red")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False
    )
    plt.ylabel("Frequency [Hz]")
    plt.tight_layout()

    plt.savefig(
        "Example_TF_Modulation.pdf",
        bbox_inches="tight",
    )


# idea: identify first the PE's that contain a certain seizure pattern

#direct frequency modulated
# RNS-1440 PE1, 2, 6, 9, 11
# RNS-1090 PE2, 3
# RNS-1603 PE3, 4, 5, 6, 7, 8

# Direct Inhibition is left out

# Fine and Coarse Fragmentation need to be annotated by Vasily

# Unaffected Seizures:
# RNS-1438 PE1
# RNS-1529 PE1, 2
# RNS-1597 PE2
# RNS-1703 PE1, 2, 3
# RNS-2543 PE1 
# RNS-6992 PE1
# RNS-7525 PE1, 2, 3
# RNS-8076 ALL
# RNS-8163 ALL
# RNS-8973 ALL
# RNS-9183 PE1, 2, 3

# Ask Vasily for 4098 PE6 and 9183 PE3 for annotation where coarse fragmentation would take place
# Ask Vasily for 1597 PE9 and 1703 PE8 for annotation where fine fragmentation would take place

# start fine fragmentation RNS-1703 PE20180627-1 time: 09:49
# start coarse fragmentation RNS-4098 PE20150804-1 time: 29:41

# Make TF Plots

# Unaffected:

PATH_UNAFFECTED = "/mnt/Nexus2/RNS_DataBank/PITT/PIT-RNS1438/iEEG/PIT-RNS1438_PE20190409-1_EOF_SZ-VK.EDF"
raw = mne.io.read_raw_edf(PATH_UNAFFECTED)

plot_TF(raw, ONSET=7144, OFFSET=7234, SZ_ON=7167, CH_IDX=0)  # schoenes Clipping Artifact

plot_TF(raw, ONSET=19685, OFFSET=19775, SZ_ON=19701, CH_IDX=0)  # schoenes Clipping Artifact


PATH_UNAFFECTED = "/mnt/Nexus2/RNS_DataBank/PITT/PIT-RNS1529/iEEG/PIT-RNS1529_PE20151118-1_EOF_SZ-VK.EDF"
raw = mne.io.read_raw_edf(PATH_UNAFFECTED)

plot_TF(raw, ONSET=5030, OFFSET=5120, SZ_ON=5073, CH_IDX=1)  # mehrere oszillationen!

plot_TF(raw, ONSET=6923, OFFSET=7013, SZ_ON=6972, CH_IDX=3)  # mehrere oszillationen, eine steigt sogar an
plot_TF(raw, ONSET=6923, OFFSET=7013, SZ_ON=6972, CH_IDX=2)
raw.plot(block=True)  

/mnt/nexus2/iESPnet/AnnotMGH/outputs/MGH-RNS0259/results

# Direct Modulation

PATH_FMOD = "/mnt/Nexus2/RNS_DataBank/PITT/PIT-RNS1440/iEEG/PIT-RNS1440_PE20150519-1_EOF_SZ-VK.EDF"
raw = mne.io.read_raw_edf(PATH_FMOD)
raw.plot(block=True)
plot_TF(raw, ONSET=2946, OFFSET=3036, SZ_ON=2972, CH_IDX=0)

plot_TF(raw, ONSET=6480, OFFSET=6570, SZ_ON=6504, CH_IDX=0)

# FINE FRAGMENTATION
PATH_FINE = "/mnt/Nexus2/RNS_DataBank/PITT/PIT-RNS1703/iEEG/PIT-RNS1703_PE20180627-1_EOF_SZ-VK.EDF"

raw = mne.io.read_raw_edf(PATH_FINE)

raw.plot(block=True)  # 09:49 second: 589: PE from 540 till 630 

# use scipy to calculate and plot stft for that one

ONSET = 540
OFFSET = 630
SZ_ON = 589
idx_ = np.logical_and(raw.times > ONSET, raw.times < OFFSET)
raw.times[idx_]
dat = raw.get_data()[:, idx_]

f, t, Zxx = signal.stft(
  dat[1, :], fs=250, nperseg=128, noverlap=125
)
z_abs = np.abs(Zxx)

plt.figure(figsize=(6,6), dpi=300)
plt.subplot(211)
plt.plot(raw.times[idx_] - raw.times[idx_][0], dat[1, :], linewidth=0.1)
plt.xlim(0, 90)
plt.subplot(212)
plt.imshow(stats.zscore(z_abs), aspect='auto')
plt.gca().invert_yaxis()
plt.axvline(x=np.where(t>(SZ_ON-ONSET))[0][0], label="Seizure Onset", color="red")


# COARSE FRAGMENTATION
PATH_FINE = "/mnt/Nexus2/RNS_DataBank/PITT/PIT-RNS4098/iEEG/PIT-RNS4098_PE20150804-1_EOF_SZ-NZ.EDF"

raw = mne.io.read_raw_edf(PATH_FINE)

raw.plot(block=True)  # 09:49 second: 589: PE from 540 till 630 

# use scipy to calculate and plot stft for that one

ONSET = 1758
OFFSET = 1818
SZ_ON = 1781
idx_ = np.logical_and(raw.times > ONSET, raw.times < OFFSET)
dat = raw.get_data()[:, idx_]

f, t, Zxx = signal.stft(
  dat[0, :], fs=250, nperseg=128, noverlap=100 #125
)
z_abs = np.abs(Zxx)


plt.imshow(stats.zscore(z_abs), aspect='auto')
plt.gca().invert_yaxis()
plt.axvline(x=np.where(t>(SZ_ON-ONSET))[0][0], label="Seizure Onset", color="red")
plt.ylim(0, 30)