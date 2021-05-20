import time
from datetime import timedelta
from utils.metrics.Metrics import Metrics

"""
 * TEI - Time Elapse Interval is a metric that meaures time elapsed per epoch for the model run
 * When initialized, start time begins
 * Each get score call uses current time minus last end time to get current elapse time
 * for this call since last recorded end time. measurement is floating point minutes
"""

class TEI(Metrics):
    def __init__(self):
        super().__init__()
        self.name = 'tei'
        self.start_time = time.time()   # init start time now
        self.end_time = self.start_time # init end time as start time

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_score(self):
        return self.tei_measure()

    def tei_measure(self):
        curr_interval = time.time()                 # grab current time
        interval = curr_interval - self.end_time    # determine interval time in seconds
        self.end_time = curr_interval               # update end time to curr_interval
        return interval / 60                        # convert to minutes
        
    
    
