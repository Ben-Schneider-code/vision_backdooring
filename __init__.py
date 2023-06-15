import os
import sys

from examples.universal_backdoor_benchmarks import benchmark_basic_poison, benchmark_binary_poison, \
    model_acc, main
from examples.universal_backdoor_embed import embed_basic_backdoor, embed_binary_enumeration_backdoor, embed_backdoor
from src.backdoor.poison.poison_label.masked_binary_enumeration_poison import DendrogramEnumerationPoison
from src.backdoor.poison.poison_label.multi_badnets import MultiBadnets
from src.backdoor.poison.poison_label.path_encoding_poison import PathEncodingPoison
from src.backdoor.poison.poison_label.unary_poison import UnaryPoison
from src.utils.hierarchical_clustering import calculate_path_encoding

if __name__ == "__main__":
    args = sys.argv

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    if args[1] == 'debug':
        calculate_path_encoding()
    elif args[1] == 'embed':
        if args[2] == 'basic_trigger':
            embed_basic_backdoor()
        if args[2] == 'binary_trigger':
            embed_binary_enumeration_backdoor()
        if args[2] == 'dendrogram':
            embed_backdoor(DendrogramEnumerationPoison)
        if args[2] == 'naive':
            embed_backdoor(MultiBadnets, patch_width=10, poison_num=250000, epochs=5)
        if args[2] == 'unary':
            embed_backdoor(UnaryPoison, patch_width=10, poison_num=75000, epochs=5)
        if args[2] == 'path':
            embed_backdoor(PathEncodingPoison, patch_width=10, poison_num=75000, epochs=5)

    elif args[1] == 'test':
        if args[2] == 'basic_trigger':
            print("benchmarking basic trigger")
            benchmark_basic_poison()
        elif args[2] == 'binary_trigger':
            print("benchmarking binary trigger")
            benchmark_binary_poison()
    elif args[1] == 'acc':
        print('testing clean model accuracy')
        model_acc()
    else:
        main()
