import time

def prettify_ms(ms):
    if (ms > 1000):
        s = int(ms / 1000)
        if (s > 60):
            # TODO: Implement this properly
            m = int(s / 60)
            ms -= 1000. * s
            s -= m * 60
            return f"{m:d} m {s:d} s {ms:.0f} ms"
        else:
            ms -= 1000 * s
            if (ms == 0):
                return f"{s:d}"
            else:
                return f"{s:d} s {ms:.0f} ms"
    else:
        return f"{ms:.3g} ms"

class Timer:
    def __init__(self):
        self.start = time.time()
    
    def get_ms(self):
        return (time.time() - self.start) * 1000

    def get_pretty(self):
        return prettify_ms(self.get_ms())

    def restart(self):
        self.start = time.time()
