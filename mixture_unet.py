import bisect
from collections import OrderedDict
from typing import Tuple

from torch import nn as nn


class MixtureOfUnets(nn.Module):
    def __init__(self, interval_models: OrderedDict[Tuple[int, int], nn.Module]):
        """
        An ensemble diffuser that applies different sub-modules depending on the current step
        :param model_intervals: a dict from (earliest and latest step)
        """
        super().__init__()
        self.intervals, self.models = zip(*interval_models.items())
        self.models = nn.ModuleList(list(self.models))
        self.config, self.add_embedding = self.models[0].config, self.models[0].add_embedding
        self.dtype = self.models[0].dtype

        prev_end = self.intervals[0][0]
        for (start, end) in self.intervals:
            assert start == prev_end
            assert end < start
            prev_end = end
        assert self.intervals[-1][1] == 0
        self.pivots = (0,) + tuple(start for start, end in self.intervals[::-1])

    def forward(self, x, t: int, **unet_kwargs):
        assert self.pivots[0] <= t < self.pivots[-1]
        index = -bisect.bisect_right(self.pivots, t)
        return self.models[index](x, t, **unet_kwargs)
