"""Bounded smooth gain envelopes."""
import math
def smooth_weight(index,start,end,fade):
    if index<start or index>=end:return 0.0
    if fade and index<start+fade:return .5-.5*math.cos(math.pi*(index-start)/fade)
    if fade and index>end-fade:return .5-.5*math.cos(math.pi*(end-index)/fade)
    return 1.0
