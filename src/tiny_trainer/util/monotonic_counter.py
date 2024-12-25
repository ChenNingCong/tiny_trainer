"""
1. self.c is growing monotonically
2. for every number n > 0 such that n % self.i == 0, we will invoke f once self.c >= n, and only once
"""
class MonotonicCounter:
    def __init__(self, i, f, invoke_zero = False):
        self.counter = 0
        self.f = f
        self.i = i
        self.c = 0
        if invoke_zero:
            f(0)
    def update(self, c):
        assert c >= self.c
        if c == self.c:
            return 
        oldc = self.c
        self.c = c
        # now we invoke f for every n in the range (oldc, c] and n % self.i == 0
        # begin is the smallest number such that begin > oldc and begin % self.i == 0
        begin = self.i * ((oldc // self.i) + 1)
        while begin <= c:
            self.f(begin)
            begin += self.i
