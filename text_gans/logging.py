class TrainingLogger:
    def __init__(self, **format):
        self.format = format

    def __call__(self, epoch, step, max_steps, **values):
        s = f'[EPOCH {epoch + 1:03d}] [{step:05d} / {len(max_steps):05d}]
