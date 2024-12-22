# coding=utf-8
import os
import argparse
import data_process
import espcn
import rdn
import rrdb
import srgan_rdn
import srgan_vanilla

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=True, help='-')
parser.add_argument('--epochs', type=int, default=100, help='-')
parser.add_argument('--upscale_factor', type=int, default=4, help='-')
parser.add_argument('--batch_size', type=int, default=2, help='-')
parser.add_argument('--lr', type=float, default=0.0001, help='-')

parser.add_argument('--rdn_num_features', type=int, default=64, help='-')
parser.add_argument('--rdn_growth_rate', type=int, default=32, help='-')
parser.add_argument('--rdn_num_blocks', type=int, default=3, help='-')
parser.add_argument('--rdn_num_layers', type=int, default=2, help='-')

parser.add_argument('--rrdb_num_features', type=int, default=64, help='-')
parser.add_argument('--rrdb_growth_rate', type=int, default=32, help='-')
parser.add_argument('--rrdb_num_blocks', type=int, default=7, help='-')
parser.add_argument('--rrdb_num_layers', type=int, default=6, help='-')

parser.add_argument('--srgan_num_features', type=int, default=64, help='-')
parser.add_argument('--srgan_num_layers', type=int, default=16, help='-')

parser.add_argument('--modelname', '-m', type=str, default='rdn', help='-')

flags, unparsed = parser.parse_known_args()


def main():
    train_loader = data_process.load_seg_data(is_train=True, batch_size=flags.batch_size)
    test_loader = data_process.load_seg_data(is_train=False, batch_size=flags.batch_size)
    if flags.modelname == 'espcn':
        model = espcn.ESPCNTrainer(flags, train_loader, test_loader)
    elif flags.modelname == 'rdn':
        model = rdn.RDNTrainer(flags, train_loader, test_loader)
    elif flags.modelname == 'rrdb':
        model = rrdb.RRDBTrainer(flags, train_loader, test_loader)
    elif flags.modelname == 'srgan':
        # model = srgan_vanilla.SRGANTrainer(flags, train_loader, test_loader)
        model = srgan_rdn.SRGANTrainer(flags, train_loader, test_loader)
    else:
        raise Exception("the model does not exist")

    model.run(mode='train')
    # model.run(mode='test')
    # model.run(mode='train and test')


if __name__ == '__main__':
    main()
