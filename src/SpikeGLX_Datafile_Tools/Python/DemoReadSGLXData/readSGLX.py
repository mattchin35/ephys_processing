# -*- coding: utf-8 -*-
"""
Requires python 3

The main() function at the bottom of this file can run from an
interpreter, or, the helper functions can be imported into a
new module or Jupyter notebook (an example is included).

Simple helper functions and python dictionary demonstrating
how to read and manipulate SpikeGLX meta and binary files.

The most important part of the demo is readMeta().
Please read the comments for that function. Use of
the 'meta' dictionary will make your data handling
much easier!

"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk
from tkinter import filedialog


# Parse ini file returning a dictionary whose keys are the metadata
# left-hand-side-tags, and values are string versions of the right-hand-side
# metadata values. We remove any leading '~' characters in the tags to match
# the MATLAB version of readMeta.
#
# The string values are converted to numbers using the "int" and "float"
# functions. Note that python 3 has no size limit for integers.
#
def readMeta(binFullPath):
    metaName = binFullPath.stem + ".meta"
    metaPath = Path(binFullPath.parent / metaName)
    metaDict = {}
    if metaPath.exists():
        # print("meta file present")
        with metaPath.open() as f:
            mdatList = f.read().splitlines()
            # convert the list entries into key value pairs
            for m in mdatList:
                csList = m.split(sep='=')
                if csList[0][0] == '~':
                    currKey = csList[0][1:len(csList[0])]
                else:
                    currKey = csList[0]
                metaDict.update({currKey: csList[1]})
    else:
        print("no meta file")
    return(metaDict)


# Return sample rate as python float.
# On most systems, this will be implemented as C++ double.
# Use python command sys.float_info to get properties of float on your system.
#
def SampRate(meta):
    if meta['typeThis'] == 'imec':
        srate = float(meta['imSampRate'])
    elif meta['typeThis'] == 'nidq':
        srate = float(meta['niSampRate'])
    elif meta['typeThis'] == 'obx':
        srate = float(meta['obSampRate'])
    else:
        print('Error: unknown stream type')
        srate = 1
        
    return(srate)


# Return a multiplicative factor for converting 16-bit file data
# to voltage. This does not take gain into account. The full
# conversion with gain is:
#         dataVolts = dataInt * fI2V / gain
# Note that each channel may have its own gain.

def Int2Volts(meta):
    if meta['typeThis'] == 'imec':
        if 'imMaxInt' in meta:
            maxInt = int(meta['imMaxInt'])
        else:
            maxInt = 512
        fI2V = float(meta['imAiRangeMax'])/maxInt
    elif meta['typeThis'] == 'nidq':
        maxInt = int(meta['niMaxInt'])
        fI2V = float(meta['niAiRangeMax'])/maxInt
    elif meta['typeThis'] == 'obx':
        maxInt = int(meta['obMaxInt'])
        fI2V = float(meta['obAiRangeMax'])/maxInt
    else:
        print('Error: unknown stream type')
        fI2V = 1
        
    return(fI2V)


# Return array of original channel IDs. As an example, suppose we want the
# imec gain for the ith channel stored in the binary data. A gain array
# can be obtained using ChanGainsIM(), but we need an original channel
# index to do the lookup. Because you can selectively save channels, the
# ith channel in the file isn't necessarily the ith acquired channel.
# Use this function to convert from ith stored to original index.
# Note that the SpikeGLX channels are 0 based.
#
def OriginalChans(meta):
    if meta['snsSaveChanSubset'] == 'all':
        # output = int32, 0 to nSavedChans - 1
        chans = np.arange(0, int(meta['nSavedChans']))
    else:
        # parse the snsSaveChanSubset string
        # split at commas
        chStrList = meta['snsSaveChanSubset'].split(sep=',')
        chans = np.arange(0, 0)  # creates an empty array of int32
        for sL in chStrList:
            currList = sL.split(sep=':')
            if len(currList) > 1:
                # each set of contiguous channels specified by
                # chan1:chan2 inclusive
                newChans = np.arange(int(currList[0]), int(currList[1])+1)
            else:
                newChans = np.arange(int(currList[0]), int(currList[0])+1)
            chans = np.append(chans, newChans)
    return(chans)


# Return counts of each nidq channel type that composes the timepoints
# stored in the binary file.
#
def ChannelCountsNI(meta):
    chanCountList = meta['snsMnMaXaDw'].split(sep=',')
    MN = int(chanCountList[0])
    MA = int(chanCountList[1])
    XA = int(chanCountList[2])
    DW = int(chanCountList[3])
    return(MN, MA, XA, DW)


# Return counts of each imec channel type that composes the timepoints
# stored in the binary files.
#
def ChannelCountsIM(meta):
    chanCountList = meta['snsApLfSy'].split(sep=',')
    AP = int(chanCountList[0])
    LF = int(chanCountList[1])
    SY = int(chanCountList[2])
    return(AP, LF, SY)

# Return counts of each obx channel type that composes the timepoints
# stored in the binary files.
#
def ChannelCountsOBX(meta):
    chanCountList = meta['snsXaDwSy'].split(sep=',')
    XA = int(chanCountList[0])
    DW = int(chanCountList[1])
    SY = int(chanCountList[2])
    return(XA, DW, SY)


# Return gain for ith channel stored in nidq file.
# ichan is a saved channel index, rather than the original (acquired) index.
#
# Note: there is nomatching function for OBX, where the gain is fixed = 1
def ChanGainNI(ichan, savedMN, savedMA, meta):
    if ichan < savedMN:
        gain = float(meta['niMNGain'])
    elif ichan < (savedMN + savedMA):
        gain = float(meta['niMAGain'])
    else:
        gain = 1    # non multiplexed channels have no extra gain
    return(gain)


# Return gain for imec channels.
# Index into these with the original (acquired) channel IDs.
#
def ChanGainsIM(meta):
    # list of probe types with NP 1.0 imro format
    np1_imro = [0,1020,1030,1200,1100,1120,1121,1122,1123,1300]
    # number of channels acquired
    acqCountList = meta['acqApLfSy'].split(sep=',')
    APgain = np.zeros(int(acqCountList[0]))     # default type = float64
    LFgain = np.zeros(int(acqCountList[1]))     # empty array for 2.0
    
    if 'imDatPrb_type' in meta:
        probeType = int(meta['imDatPrb_type'])
    else:
        probeType = 0
    
    if sum(np.isin(np1_imro, probeType)):
        # imro + probe allows setting gain independently for each channel
        imroList = meta['imroTbl'].split(sep=')')
        # One entry for each channel plus header entry,
        # plus a final empty entry following the last ')'        
        for i in range(0, int(acqCountList[0])):
            currList = imroList[i+1].split(sep=' ')
            APgain[i] = float(currList[3])
            LFgain[i] = float(currList[4])
    else:                 
        # get gain from imChan0apGain
        if 'imChan0apGain' in meta:
            APgain = APgain + float(meta['imChan0apGain'])
            if int(acqCountList[1]) > 0:                
                LFgain = LFgain + float(meta['imChan0lfGain'])
        elif (probeType == 1110):
            # active UHD, for metadata lacking imChan0apGain, get gain from
            # imro table header
            imroList = meta['imroTbl'].split(sep=')')
            currList = imroList[0].split(sep=',')
            APgain = APgain + float(currList[3])
            LFgain = LFgain + float(currList[4])
        elif (probeType == 21) or (probeType == 24):
            # development NP 2.0; APGain = 80 for all AP
            # return 0 for LFgain (no LF channels)
            APgain = APgain + 80        
        elif (probeType == 2013):
            # commercial NP 2.0; APGain = 100 for all AP
            APgain = APgain + 100
        else:
            print('unknown gain, setting APgain to 1')
            APgain = APgain + 1
    fI2V = Int2Volts(meta)
    APChan0_to_uV = 1e6*fI2V/APgain[0]
    if LFgain.size > 0:
        LFChan0_to_uV = 1e6*fI2V/LFgain[0]
    else:
        LFChan0_to_uV = 0
    return(APgain, LFgain, APChan0_to_uV, LFChan0_to_uV)


# Having accessed a block of raw nidq data using makeMemMapRaw, convert
# values to gain-corrected voltage. The conversion is only applied to the
# saved-channel indices in chanList. Remember, saved-channel indices are
# in the range [0:nSavedChans-1]. The dimensions of dataArray remain
# unchanged. ChanList examples:
# [0:MN-1]  all MN channels (MN from ChannelCountsNI)
# [2,6,20]  just these three channels (zero based, as they appear in SGLX).
#
def GainCorrectNI(dataArray, chanList, meta):
    MN, MA, XA, DW = ChannelCountsNI(meta)
    fI2V = Int2Volts(meta)
    # print statements used for testing...
    # print("NI fI2V: %.3e" % (fI2V))
    # print("NI ChanGainNI: %.3f" % (ChanGainNI(0, MN, MA, meta)))

    # make array of floats to return. dataArray contains only the channels
    # in chanList, so output matches that shape
    convArray = np.zeros(dataArray.shape, dtype=float)
    for i in range(0, len(chanList)):
        j = chanList[i]             # index in saved data
        conv = fI2V/ChanGainNI(j, MN, MA, meta)
        # dataArray contains only the channels in chanList
        convArray[i, :] = dataArray[i, :] * conv
    return(convArray)

# Having accessed a block of raw obx data using makeMemMapRaw, convert
# values to volts. The conversion is only applied to the
# saved-channel indices in chanList. Remember, saved-channel indices are
# in the range [0:nSavedChans-1]. The dimensions of dataArray remain
# [2,6,20]  just these three channels (zero based, as they appear in SGLX).
#
def GainCorrectOBX(dataArray, chanList, meta):

    fI2V = Int2Volts(meta)

    # make array of floats to return. dataArray contains only the channels
    # in chanList, so output matches that shape
    convArray = np.zeros(dataArray.shape, dtype=float)
    for i in range(0, len(chanList)):
        # dataArray contains only the channels in chanList
        convArray[i, :] = dataArray[i, :] * fI2V
    return(convArray)


# Having accessed a block of raw imec data using makeMemMapRaw, convert
# values to gain corrected voltages. The conversion is only applied to
# the saved-channel indices in chanList. Remember saved-channel indices
# are in the range [0:nSavedChans-1]. The dimensions of the dataArray
# remain unchanged. ChanList examples:
# [0:AP-1]  all AP channels
# [2,6,20]  just these three channels (zero based)
# Remember that for an lf file, the saved channel indices (fetched by
# OriginalChans) will be in the range 384-767 for a standard 3A or 3B probe.
#
def GainCorrectIM(dataArray, chanList, meta):
    # Look up gain with acquired channel ID
    chans = OriginalChans(meta)
    APgain, LFgain, _, _ = ChanGainsIM(meta)
    nAP = len(APgain)
    nNu = nAP * 2

    # Common conversion factor
    fI2V = Int2Volts(meta)

    # make array of floats to return. dataArray contains only the channels
    # in chanList, so output matches that shape
    convArray = np.zeros(dataArray.shape, dtype='float')
    for i in range(0, len(chanList)):
        j = chanList[i]     # index into timepoint
        k = chans[j]        # acquisition index
        if k < nAP:
            conv = fI2V / APgain[k]
        elif k < nNu:
            conv = fI2V / LFgain[k - nAP]
        else:
            conv = 1
        # The dataArray contains only the channels in chanList
        convArray[i, :] = dataArray[i, :]*conv
    return(convArray)

# Return memmap for the raw data
# Fortran ordering is used to match the MATLAB version
# of these tools.
#
def makeMemMapRaw(binFullPath, meta, allowWrite=False):
    nChan = int(meta['nSavedChans'])
    nFileSamp = int(int(meta['fileSizeBytes'])/(2*nChan))
    if allowWrite:
        print("Warning: write access allowed!!!")
        modeStr = 'r+'
    else:
        modeStr = 'r'

    print("nChan: %d, nFileSamp: %d" % (nChan, nFileSamp))
    rawData = np.memmap(binFullPath, dtype='int16', mode=modeStr,
                        shape=(nChan, nFileSamp), offset=0, order='F')
    return rawData


# Return an array [lines X timepoints] of uint8 values for a
# specified set of digital lines.
#
# - dwReq is the zero-based index into the saved file of the
#    16-bit word that contains the digital lines of interest.
# - dLineList is a zero-based list of one or more lines/bits
#    to scan from word dwReq.
#
def ExtractDigital(rawData, firstSamp, lastSamp, dwReq, dLineList, meta):
    # Get channel index of requested digital word dwReq
    if meta['typeThis'] == 'imec':
        AP, LF, SY = ChannelCountsIM(meta)
        if SY == 0:
            print("No imec sync channel saved.")
            digArray = np.zeros((0), 'uint8')
            return digArray
        else:
            digCh = AP + LF + dwReq
    elif meta['typeThis'] == 'nidq':
        MN, MA, XA, DW = ChannelCountsNI(meta)
        if dwReq > DW-1:
            print("Maximum digital word in file = %d" % (DW-1))
            digArray = np.zeros((0), 'uint8')
            return digArray
        else:
            digCh = MN + MA + XA + dwReq
    elif meta['typeThis'] == 'obx':
        XA, DW, SY = ChannelCountsOBX(meta)
        if dwReq > DW-1:
            print("Maximum digital word in file = %d" % (DW-1))
            digArray = np.zeros((0), 'uint8')
            return digArray
        else:
            digCh = XA + dwReq
    else:
        print('unknown data stream')

    selectData = np.ascontiguousarray(rawData[digCh, firstSamp:lastSamp+1], 'int16')
    nSamp = lastSamp-firstSamp + 1

    # unpack bits of selectData; unpack bits works with uint8
    # original data is int16
    bitWiseData = np.unpackbits(selectData.view(dtype='uint8'))
    # output is 1-D array, nSamp*16. Reshape and transpose
    bitWiseData = np.transpose(np.reshape(bitWiseData, (nSamp, 16)))

    nLine = len(dLineList)
    digArray = np.zeros((nLine, nSamp), 'uint8')
    for i in range(0, nLine):
        byteN, bitN = np.divmod(dLineList[i], 8)
        targI = byteN*8 + (7 - bitN)
        digArray[i, :] = bitWiseData[targI, :]

    return digArray


# Sample calling program to get a file from the user,
# read metadata fetch sample rate, voltage conversion
# values for this file and channel, and plot a small range
# of voltages from a single channel.
# Note that this code merely demonstrates indexing into the
# data file, without any optimization for efficiency.
#
def main():

    # Get file from user
    root = Tk()         # create the Tkinter widget
    root.withdraw()     # hide the Tkinter root window

    # Windows specific; forces the window to appear in front
    root.attributes("-topmost", True)

    binFullPath = Path(filedialog.askopenfilename(title="Select binary file"))
    root.destroy()      # destroy the Tkinter widget

    # Other parameters about what data to read
    tStart = 0
    tEnd = 2
    dataType = 'A'    # 'A' for analog, 'D' for digital data

    # For analog channels: zero-based index of a channel to extract,
    # gain correct and plot (plots first channel only)
    chanList = [0]

    # For a digital channel: zero based index of the digital word in
    # the saved file. For imec data there is never more than one digital word.
    dw = 0

    # Zero-based Line indices to read from the digital word and plot.
    # For 3B2 imec data: the sync pulse is stored in line 6.
    dLineList = [0, 1, 6]

    # Read in metadata; returns a dictionary with string for values
    meta = readMeta(binFullPath)

    # parameters common to NI and imec data
    sRate = SampRate(meta)
    firstSamp = int(sRate*tStart)
    lastSamp = int(sRate*tEnd)
    # array of times for plot
    tDat = np.arange(firstSamp, lastSamp+1, dtype='uint64')
    tDat = 1000*tDat/sRate      # plot time axis in msec

    rawData = makeMemMapRaw(binFullPath, meta)

    if dataType == 'A':
        selectData = rawData[chanList, firstSamp:lastSamp+1]
        if meta['typeThis'] == 'imec':
            # apply gain correction and convert to uV
            convData = 1e6*GainCorrectIM(selectData, chanList, meta)
        elif meta['typeThis'] == 'nidq':
            MN, MA, XA, DW = ChannelCountsNI(meta)
            # print("NI channel counts: %d, %d, %d, %d" % (MN, MA, XA, DW))
            # apply gain correction and convert to mV
            convData = 1e3*GainCorrectNI(selectData, chanList, meta)
        elif meta['typeThis'] == 'obx':
            # Gain correct is just conversion to volts           
            convData = 1e3*GainCorrectOBX(selectData, chanList, meta)
            
        # Plot the first of the extracted channels
        fig, ax = plt.subplots()
        ax.plot(tDat, convData[0, :])
        plt.show()

    else:
        digArray = ExtractDigital(rawData, firstSamp, lastSamp, dw,
                                  dLineList, meta)

        # Plot the first of the extracted channels
        fig, ax = plt.subplots()

        for i in range(0, len(dLineList)):
           ax.plot(tDat, digArray[i, :])
        plt.show()


if __name__ == "__main__":
    main()
