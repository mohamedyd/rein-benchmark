import time
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GlobalTimer(object):

    def __init__(self,log=True):
        self.log = log
        if log:
            self.time_log = []
        self.origin = time.time()
        self.start = self.origin

    def time_point(self,msg):
        curr = time.time()
        time_pt=curr-self.origin
        info = "[{time_pt}] {msg}\n".format(time_pt=time_pt,msg=msg)
        if self.log:
            self.time_log.append([time_pt,msg,0])
        logger.info(info)

    def time_start(self,msg):
        curr = time.time()
        time_pt=curr-self.origin
        info = "[{time_pt}] {msg} start\n".format(time_pt=time_pt,msg=msg)
        if self.log:
            self.time_log.append([time_pt,"{} start".format(msg),0])
        self.start = curr
        logger.info(info)

    def time_end(self,msg):
        curr = time.time()
        time_pt=curr-self.origin
        exec_time = curr - self.start
        info = "[{time_pt}] {msg} execution time: {t}\n".format(time_pt=time_pt,msg=msg, t=exec_time)
        if self.log:
            self.time_log.append([time_pt,"end: {}".format(msg),exec_time])
        self.start = curr
        logger.info(info)
        return exec_time

    def to_file(self):
        log = pd.DataFrame(data=self.time_log,columns=['time_point','msg','execution_time'])
        log.to_csv("time_points.csv", index=False)
