import time
from collections import deque
class Timer():
    def __init__(self, maxitem=10):
        self.ticks = deque()
        self.maxitem = 10
    def step(self):
        self.ticks.append(time.time())
        while len(self.ticks) > self.maxitem:
            self.ticks.popleft()
    def rate(self):
        if len(self.ticks) <= 1:
            return 0
        else:
            l = self.ticks[0]
            r = self.ticks[-1]
            return (len(self.ticks) - 1) / (r - l) 