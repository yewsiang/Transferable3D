# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import time

class Timer(object):
    """A simple timer."""
    def __init__(self, name=None):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.name = name

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True, print_time=True, prepend_str=""):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        prepend_str = "" if (prepend_str == "") else ("[%s] " % prepend_str)
        if print_time and average:
        	if self.name is None:
        		print("%sAverage time taken: %.2f s" % (prepend_str, self.average_time))
        	else:
        		print("%sAverage time taken for %s: %.2f s" % (prepend_str, self.name, self.average_time))
        elif print_time:
        	if self.name is None:
        		print("%sTime taken: %.2f s" % (prepend_str, self.diff))
        	else:
        		print("%sTime taken for %s: %.2f s" % (prepend_str, self.name, self.diff))
        if average:
            return self.average_time
        else:
            return self.diff
