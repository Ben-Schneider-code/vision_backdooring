import os

from examples.universal_backdoor_benchmarks import load_and_bench

if __name__ == "__main__":
     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
     load_and_bench()
