class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WarmUp(object):
    "Returns the value of the anneling factor to be used"

    def __init__(self, epochs=100, value=1.0):
        self.epoch = 0
        self.max_epoch = epochs
        self.value = value

    def get(self):

        if self.epoch >= self.max_epoch:
            return self.value
        else:
            return self.value*(float(self.epoch)/self.max_epoch)

    def update(self):
        self.epoch += 1



class WarmUpStep(object):
    "Returns the value of the anneling factor to be used"

    def __init__(self, epochs=100, value=1.0):
        self.epoch = 0
        self.max_epoch = epochs
        self.value = value

    def get(self):

        if self.epoch >= self.max_epoch:
            return self.value
        else:
            return 0.0

    def update(self):
        self.epoch += 1