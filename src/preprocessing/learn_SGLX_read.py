# This example imports functions from the DemoReadSGLXData module to read
# digital data. The metadata file must be present in the same directory as the binary file.
# Works with both imec and nidq digital channels.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.DemoReadSGLXData.readSGLX import readMeta, SampRate, makeMemMapRaw, ExtractDigital, GainCorrectIM, GainCorrectNI

data_root = Path.home().joinpath('Documents', 'testdata')
ibl_bin = data_root.joinpath('IBL_testdata', 'imec_385_100s.ap.bin')
rec_ap_bin = data_root.joinpath('HD015_11.30.2023/HD015_11.30.2023_g0/HD015_11.30.2023_g0_imec0/HD015_11.30.2023_g0_t0.imec0.ap.bin')
rec_lf_bin = data_root.joinpath('HD015_11.30.2023/HD015_11.30.2023_g0/HD015_11.30.2023_g0_imec0/HD015_11.30.2023_g0_t0.imec0.lf.bin')

binary_file = rec_ap_bin

# Other parameters about what data to read
tStart = 0        # in seconds
tEnd = 1

# Which digital word to read.
# For imec, there is only 1 digital word, dw = 0.
# For NI, digital lines 0-15 are in word 0, lines 16-31 are in word 1, etc.
dw = 0

# Which lines within the digital word, zero-based
# Note that the SYNC line for PXI 3B is stored in line 6.
dLineList = [0,1,6]
meta = readMeta(binary_file)
sRate = SampRate(meta)
sRate = int(sRate)  # round to nearest integer; SGLX records calibrated rate but that's not necessary here
print("Sampling rate of {} is {} Hz".format(binary_file.name, sRate))

firstSamp = int(sRate*tStart)
lastSamp = int(sRate*tEnd)
rawData = makeMemMapRaw(binary_file, meta)


### DIGITAL DATA ###
# get digital data for the selected lines
digArray = ExtractDigital(rawData, firstSamp, lastSamp, dw, dLineList, meta)

# Plot the extracted digital channels
tDat = np.arange(firstSamp, lastSamp+1)
tDat = 1000*tDat/sRate      # plot time axis in msec
fig_d, ax_d = plt.subplots()
for i in range(0, len(dLineList)):
    ax_d.plot(tDat, digArray[i, :])
plt.ylabel('Digital line value')
plt.xlabel('Time (msec)')
# plt.show()


### ANALOG DATA ###

# Other parameters about what data to read
tStart = 0        # in seconds
tEnd = 0.1
chanList = [0, 10]    # list of channels to extract, by index in saved file
firstSamp = int(sRate*tStart)
lastSamp = int(sRate*tEnd)

selectData = rawData[chanList, firstSamp:lastSamp+1]
if meta['typeThis'] == 'imec':
    # apply gain correction and convert to uV
    convData = 1e6*GainCorrectIM(selectData, chanList, meta)
else:
    # apply gain correction and convert to mV
    convData = 1e3*GainCorrectNI(selectData, chanList, meta)

# Plot the first of the extracted channels
tDat = np.arange(firstSamp, lastSamp+1)
tDat = 1000*tDat/sRate      # plot time axis in msec
fig_a, ax_a = plt.subplots()
# for i, chan in enumerate(chanList):
i = 1
ax_a.plot(tDat, convData[i])
plt.ylabel('Voltage (uV)')
plt.xlabel('Time (msec)')
plt.title('Channel {}'.format(chanList[i]))


plt.show()

