# coding=utf-8
import os
import argparse
import ws_data_process_2
import ws_wespe_2

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=True, help='-')
parser.add_argument('--epochs', type=int, default=5, help='-')
parser.add_argument('--upscale_factor', type=int, default=4, help='-')
parser.add_argument('--batch_size', type=int, default=8, help='-')
parser.add_argument('--lr', type=float, default=0.0002, help='-')

parser.add_argument('--num_features', type=int, default=64, help='-')
parser.add_argument('--growth_rate', type=int, default=32, help='-')
parser.add_argument('--num_blocks', type=int, default=3, help='-')
parser.add_argument('--num_layers', type=int, default=2, help='-')

# parser.add_argument('--srgan_g_save', type=str, default='srgan_G_params_1.pkl', help='-')
# parser.add_argument('--srgan_d_save', type=str, default='srgan_D_params_1.pkl', help='-')
parser.add_argument('--modelname', '-m', type=str, default='wespe', help='-')

flags, unparsed = parser.parse_known_args()


def main():
    train_loader, test_loader = ws_data_process_2.load_seg_data(batch_size=flags.batch_size)
    if flags.modelname == 'wespe':
        model = ws_wespe_2.WESPE2Trainer(flags, train_loader, test_loader)
    else:
        raise Exception("the model does not exist")

    model.run()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()
