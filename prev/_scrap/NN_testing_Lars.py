# plotting
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# animation
from JSAnimation.IPython_display import display_animation
from matplotlib import animation

# matrices
import numpy as np
import scipy.sparse as sparsem
import scipy.sparse.linalg as sparsela

# JavaScript object notation
import json

# bandpass filtering, should be in tarball
import bandpass as bp

def addBias(m):
    """Add a column of ones for a bias term.
    """
    m1 = np.empty((m.shape[0], m.shape[1]+1))
    m1[:,:-1] = m
    m1[:,-1] = 1.0
    return m1


def loadData(fileName='s20-gammasys-gifford-unimpaired.json'):
    """Load some resting state EEG d_rec.
    """
    dataHandle = open(fileName, 'r')
    data = json.load(dataHandle)
    dataHandle.close()

    data = data[0]  # pull out resting state session
    sampRate = data['sample rate']
    chanNames = data['channels']
    nChan = len(chanNames)
    sig = np.array(data['eeg']['trial 1']).T
    sig = sig[:, :nChan]  # omit trigger channel

    return sig, chanNames, sampRate

if False:
    # sinusoid d_rec for testing
    time = np.arange(0.0, 20.0*np.pi, 0.1)[:,None]
    sig = np.sin(time) #+ np.random.normal(scale=0.15, size=tFull.shape)
    window = np.arange(int((4.0*np.pi)/0.1))
else:
    # EEG d_rec
    sig, chanNames, sampRate = loadData()
    filt = bp.IIRFilter(1.0, 12.0, sampRate=sampRate, order=3)
    sig = filt.filter(sig)
    sig = sig[:(10.0*sampRate),:]
    time = np.arange(0.0, 10.0, 1.0/sampRate)[:,None]
    winWidth = 2.0
    window = np.arange(int(winWidth*sampRate))

# for separating signal channels
scale = np.max(np.abs(sig))
sep = np.arange(sig.shape[1]) * 2.0 * scale

fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(1,1,1)
ax.plot(time, sig+sep[None,:])
ax.set_yticks(sep)
ax.set_yticklabels(chanNames)
ax.set_xlabel('Time (s)')
ax.autoscale(tight=True)

# zoomed inset plot
subx1,subx2 = (0.0,winWidth)
#suby1,suby2 = (-scale,sep[-1]+scale)
suby1,suby2 = (-scale,scale)
axins = zoomed_inset_axes(ax, zoom=4.0,
            bbox_to_anchor=(-1.0,sep[-1]),
            bbox_transform=ax.transData)
axins.plot(time[window], sig[window]+sep[None,:])

axins.set_xlim(subx1,subx2)
axins.set_ylim(suby1,suby2)
axins.get_xaxis().set_visible(False)
axins.get_yaxis().set_visible(False)
mark_inset(ax,axins, loc1=3, loc2=1, fc="none", ec="0.5");