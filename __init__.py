import os
import sys

from examples.universal_backdoor_benchmarks import benchmark_basic_poison, benchmark_binary_enumeration_poison, \
    model_acc, main
from examples.universal_backdoor_embed import embed_basic_backdoor, embed_binary_enumeration_backdoor

if __name__ == "__main__":
    args = sys.argv

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    if args[1] == 'embed':
        if args[2] == 'basic_trigger':
            embed_basic_backdoor()
        if args[2] == 'binary_trigger':
            embed_binary_enumeration_backdoor()

    elif args[1] == 'test':
        if args[2] == 'basic_trigger':
            print("benchmarking basic trigger")
            benchmark_basic_poison()
        elif args[2] == 'binary_trigger':
            print("benchmarking binary trigger")
            benchmark_binary_enumeration_poison()
    elif args[1] == 'acc':
        print('testing clean model accuracy')
        model_acc()
    else:
        main()
