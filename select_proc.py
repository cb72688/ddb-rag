import torch


def detect_proc():
    if torch.cuda.is_available():
        return "cuda"
    elif not torch.cuda.is_available():
        return "cpu"
    else:
        print(f"{__file__} :: {__name__} :: {
              __package__} -- Error determining which processor is available and used for inference.")
        return "ERR"
