"""
Create sample pulse sequences for synchronization testing.

1. Base 30kHz, target 30kHz, no dropped frames
2. Base 30kHz, target 5kHz, no dropped frames
3. Base 30kHz, target 5kHz, 1 dropped frame

Also test when the base pulse has more pulses than the target pulse because it started and ended later.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from icecream import ic

rng = np.random.default_rng(0)

class SyncPulse:

    def __init__(self, end_time):
        self.t = 0
        self.ix = 0
        self.end_time = end_time
        self.state = 0
        self.delay = 0  # delay to next state change
        self.signal = []
        self.timestamp = []

        self.jitter = .2
        self.t_high = .5
        self.t_low = .5

    def step(self, dt: float) -> None:
        self.delay -= dt
        self.t += dt
        self.ix += 1
        return

    def get_state(self) -> int:
        if self.delay <= 0:
            self.state = 1 - self.state
            self.signal.append(self.state)
            self.timestamp.append(self.t)
            if self.state == 1:
                self.delay = self.t_high + self.jitter * rng.random() * rng.choice([-1, 1]) - np.abs(self.delay)
            else:
                self.delay = self.t_low - np.abs(self.delay) # + self.jitter * rng.random() * rng.choice([-1, 1])
        return self.state


class SignalReader:

    def __init__(self, sampling_rate: float, p_framedrop=0):
        self.t = 0
        self.ix = 0
        self.sampling_rate = sampling_rate
        self.delay = 0  # delay to reading next sample
        self.signal = []
        self.timestamp = []
        self.p_framedrop = 0  # probability of dropping a frame

    def step(self, dt: float) -> None:
        self.delay -= dt
        self.t += dt
        self.ix += 1
        return

    def read(self, signal: SyncPulse) -> None:
        if self.delay <= 0:
            self.delay = 1 / self.sampling_rate - np.abs(self.delay)
            if rng.random() > self.p_framedrop:
                self.signal.append(signal.get_state())
                self.timestamp.append(self.t)
            else:
                pass
        return


def main():
    end_time = 10
    dt = 1/30000
    sync = SyncPulse(end_time=end_time)
    electrodes = SignalReader(sampling_rate=30000)
    daq = SignalReader(sampling_rate=10000)  # start with the electrodes, no frame drops
    camera = SignalReader(sampling_rate=5000)  # start before the behavior box, random frame drops
    behavbox = SignalReader(sampling_rate=5000) # start last, no frame drops

    while sync.t < end_time:
        sync.step(dt)
        electrodes.step(dt)
        daq.step(dt)
        camera.step(dt)
        behavbox.step(dt)

        electrodes.read(sync)
        daq.read(sync)
        camera.read(sync)
        behavbox.read(sync)

        if sync.ix % 1000 == 0:
            ic('time elapsed:', sync.t)

    f, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(sync.timestamp, sync.signal, label='sync')
    ax[1].plot(electrodes.timestamp, electrodes.signal, label='electrodes')
    ax[2].plot(daq.timestamp, daq.signal, label='daq')
    ax[3].plot(camera.timestamp, camera.signal, label='camera')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
