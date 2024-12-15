# coding=utf-8
import os
import argparse
import ws_data_process_1
import ws_wespe_1

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=True, help='-')
parser.add_argument('--epochs', type=int, default=5, help='-')
parser.add_argument('--upscale_factor', type=int, default=4, help='-')
parser.add_argument('--batch_size', type=int, default=2, help='-')
parser.add_argument('--lr', type=float, default=0.0002, help='-')
parser.add_argument('--modelname', '-m', type=str, default='wespe', help='-')

flags, unparsed = parser.parse_known_args()


def main():
    train_loader, test_loader = ws_data_process_1.load_seg_data(batch_size=flags.batch_size)
    if flags.modelname == 'wespe':
        model = ws_wespe_1.WESPETrainer(flags, train_loader, test_loader)
    else:
        raise Exception("the model does not exist")

    model.run()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()
