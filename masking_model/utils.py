from pathlib import Path

def convert_to_path(*args) -> list[Path]:
    paths = []
    for p in args:
        if p is None: paths.append(None)
        else: paths.append(Path(p))

    return paths



class AverageCalculator:
    def __init__(self):
        self.reset()

    def update(self, num, count=1):
        self.count += count
        self.sum += num * count

    def avg(self):
        return self.sum/self.count

    def reset(self):
        self.sum = self.count = 0.0
