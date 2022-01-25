import numpy as np
from collections import deque


class FrameBuffer:
    """A circular buffer implemented as a deque to keep track of the last few
    frames in the environment that together form a state capturing temporal
    and directional information. Provides an accessor to get the current
    state at any given time, which is represented as a list of consecutive
    frames.

    Also takes pre/post-processors to potentially resize or modify the frames
    before inserting them into the buffer."""

    def __init__(self,
                 frames_per_state,
                 preprocessor=lambda x: x,
                 postprocessor=lambda x: np.concatenate(x, dim=1)):
        """
        @param frames_per_state:  Number of consecutive frames that form a state.
        @param preprocessor:      Lambda that takes a frame and returns a
                                  preprocessed frame.
        """
        if frames_per_state <= 0:
            raise RuntimeError('Frames per state should be greater than 0')

        self.frames_per_state = frames_per_state
        self.samples = deque(maxlen=frames_per_state)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def append(self, sample):
        """
        Takes a frame, applies preprocessing, and appends it to the deque.
        The first frame added to the buffer is duplicated `frames_per_state` times
        to completely fill the buffer.
        """
        sample = self.preprocessor(sample)
        if len(self.samples) == 0:
            self.samples.extend(self.frames_per_state * [sample])
        self.samples.append(sample)

    def get_state(self):
        """
        Fetch the current state consisting of `frames_per_state` consecutive frames.
        If `frames_per_state` is 1, returns the frame instead of an array of
        length 1. Otherwise, returns a Numpy array of `frames_per_state` frames.
        """
        if len(self.samples) == 0:
            return None
        if self.frames_per_state == 1:
            return self.postprocessor([self.samples[0]])
        return self.postprocessor(list(self.samples))

    def reset(self):
        self.samples.clear()
