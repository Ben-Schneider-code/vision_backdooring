import os
from examples.universal_backdoor_embed import embed_universal_backdoor

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    embed_universal_backdoor()
