# Stand-in: experiment tracking is off (logging.report_to: none).
run = None

def log(*args, **kwargs):
    return None

def init(*args, **kwargs):
    raise RuntimeError("wandb is not installed; set logging.report_to: none")

def finish(*args, **kwargs):
    return None
