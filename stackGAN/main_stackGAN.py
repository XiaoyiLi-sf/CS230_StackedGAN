"""
Author: Yuejey Choi, Minje Choi, Munyong Kim, Jung-Woo Ha, Sung Kim, Jaegul Choo
Modifier: Xiaoyi Li and Xiaowen Yu
"""
import os
import argparse
import gc
from solver_stackGAN_new import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.intermediate_dir):
        os.makedirs(config.intermediate_dir)
    if not os.path.exists(config.result_dir_starGAN):
        os.makedirs(config.result_dir_starGAN)
           

    # Data loader.
    rafdb_loader = get_loader(config.rafdb_image_dir, config.image_size, config.batch_size_starGAN,
                                 config.mode_starGAN, config.num_workers)

    # Solver for training and testing StarGAN.
    solver = Solver(rafdb_loader, config)

    if (config.mode_starGAN == 'train') and (config.mode_cartoonGAN == 'full'):
        solver.pretrain_generator()
        gc.collect()
        solver.train()
    elif (config.mode_starGAN == 'train') and (config.mode_cartoonGAN == 'pretrain'):
        solver.pretrain_generator()
        solver.train()
    elif (config.mode_starGAN == 'train') and (config.mode_cartoonGAN == 'gan'):
        solver.train()      
    elif config.mode_starGAN == 'test':
        solver.test()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    "for StarGAN"
    # Model configuration.
    parser.add_argument('--stack_mode', type=str, default='A', choices=['A','B'])
    parser.add_argument('--c_dim', type=int, default=7, help='dimension of domain labels')
    parser.add_argument('--image_size', type=int, default=50, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--batch_size_starGAN', type=int, default=4, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=50000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=25000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode_starGAN', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--rafdb_image_dir', type=str, default='datasets/RaFDB/train')
    parser.add_argument('--log_dir', type=str, default='stargan_rafdb/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan_rafdb/models')
    parser.add_argument('--intermediate_dir', type=str, default='datasets/cartoon_data/trainA')
    parser.add_argument('--sample_dir', type=str, default='stargan_rafdb/samples')
    parser.add_argument('--result_dir_starGAN', type=str, default='stargan_rafdb/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    "for CartoonGAN"
    parser.add_argument("--mode_cartoonGAN", type=str, default="full",
                        choices=["full", "pretrain", "gan"])
    parser.add_argument("--dataset_name", type=str, default="realworld2cartoon")
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--batch_size_cartoonGAN", type=int, default=1)
    parser.add_argument("--sample_size", type=int, default=8)
    parser.add_argument("--source_domain", type=str, default="A")
    parser.add_argument("--target_domain", type=str, default="B")
    parser.add_argument("--gan_type", type=str, default="lsgan", choices=["gan", "lsgan"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--reporting_steps", type=int, default=100)
    parser.add_argument("--content_lambda", type=float, default=10)
    parser.add_argument("--style_lambda", type=float, default=1.)
    parser.add_argument("--g_adv_lambda", type=float, default=1)
    parser.add_argument("--d_adv_lambda", type=float, default=1)
    parser.add_argument("--generator_lr", type=float, default=1e-5)
    parser.add_argument("--discriminator_lr", type=float, default=1e-5)
    parser.add_argument("--ignore_vgg", action="store_true")
    parser.add_argument("--pretrain_learning_rate", type=float, default=1e-5)
    parser.add_argument("--pretrain_epochs", type=int, default=2)
    parser.add_argument("--pretrain_saving_epochs", type=int, default=1)
    parser.add_argument("--pretrain_reporting_steps", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--log_dir_cartoonGAN", type=str, default="runs")
    parser.add_argument("--result_dir_cartoonGAN", type=str, default="result")
    parser.add_argument("--checkpoint_dir", type=str, default="training_checkpoints")
    parser.add_argument("--generator_checkpoint_prefix", type=str, default="generator")
    parser.add_argument("--discriminator_checkpoint_prefix", type=str, default="discriminator")
    parser.add_argument("--pretrain_checkpoint_prefix", type=str, default="pretrain_generator")
    parser.add_argument("--pretrain_model_dir", type=str, default="models")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--disable_sampling", action="store_true")
    # TODO: rearrange the order of options
    parser.add_argument(
        "--pretrain_generator_name", type=str, default="pretrain_generator"
    )
    parser.add_argument("--generator_name", type=str, default="generator")
    parser.add_argument("--discriminator_name", type=str, default="discriminator")
    parser.add_argument("--not_show_progress_bar", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show_tf_cpp_log", action="store_true")


    config = parser.parse_args()
    print(config)
    main(config)