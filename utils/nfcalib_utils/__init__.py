import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Easy args NFlib')
    parser.add_argument('--data', action="store", type=str)
    parser.add_argument('--model', action="store", type=str)
    parser.add_argument('--index', action='store', type=int, default='1')
    parser.add_argument('--num_layers', action="store", type=int, default=5)
    parser.add_argument('--num_epoch', action="store", type=int, default=int(5 * 1e5))
    parser.add_argument('--lr', action="store", type=float, default=0.0005)
    parser.add_argument('--checkpoint_frequency', action="store", type=int, default=int(1 * 1e5))
    args = parser.parse_args()
    return args



