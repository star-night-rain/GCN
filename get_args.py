import torch
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter

def get_args():
    parser = ArgumentParser('GCN',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser.add_argument('--device',default=device)
    parser.add_argument('--seed',default=42)
    parser.add_argument('--epochs',default=200)
    parser.add_argument('--lr',default=0.01)
    parser.add_argument('--dropout',default=0.5)

    parser.add_argument('--num_layers',default=2)
    parser.add_argument('--input_dim',default=1433)
    parser.add_argument('--hidden_dim',default=16)
    parser.add_argument('--output_dim',default=7)
    parser.add_argument('--weight_decay',default=5e-4)

    args = parser.parse_args()
    return args