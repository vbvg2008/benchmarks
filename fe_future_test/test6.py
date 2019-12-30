class Estimator:
    def __init__(self, trace):
        self.system = System(mode="train", global_step=0, num_devices=1, log_steps=100, total_epochs=10, total_steps=10000, epoch_idx=0, batch_idx=0)
        self.trace = trace
        self.trace.system = self.system

    def loop_through(self):
        for self.system.epoch_idx in range(10):
            print(self.system.epoch_idx)
            print(self.trace.system.epoch_idx)
class System:
    def __init__(self, mode, global_step, num_devices, log_steps, total_epochs, total_steps, epoch_idx, batch_idx):
        self.mode = mode
        self.global_step = global_step
        self.num_devices = num_devices
        self.log_steps = log_steps
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.epoch_idx = epoch_idx
        self.batch_idx = batch_idx
        self.buffer = {}

    def add_buffer(self, key, value):
        self.buffer[key] = value

    def clear_buffer(self):
        del self.buffer
        self.buffer = {}


class Trace:
    def __init__(self):
        self.system = None

if __name__ == "__main__":
    est = Estimator(trace=Trace())
    est.loop_through()