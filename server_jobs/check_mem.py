import gc, torch
gc.collect()
torch.cuda.empty_cache()
print("GPU 0 allocated GiB:", round(torch.cuda.memory_allocated(0)/(1024**3), 3))
print("GPU 0 reserved GiB:", round(torch.cuda.memory_reserved(0)/(1024**3), 3))
RESULT = "DONE"

