import argparse

def str2bool(v):
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():

    parser = argparse.ArgumentParser()

    # Regression loss
    parser.add_argument('--reg_loss_type', type = int, choices = [1,2], default = 1,
                    help = 'Degree of Norm for regression loss. One of L1 or L2.')
    parser.add_argument('--d_input', type = str, choices = ['pose', 'motion', 'both'], default = 'motion',
                    help = 'Input to discriminator. One of pose, motion or both.')
    parser.add_argument('--lambda_motion_reg_loss', type = float, default = 1,
                    help = 'Multiplier for motion reg loss in case both motion and pose are used')
    # Encoder loss
    parser.add_argument('--margin', type = float, default = 0.5,
                    help = 'Margin in triplet loss for encoder')
    parser.add_argument('--enc_loss_type', type = int, choices = [1,2], default = 1,
                    help = 'Degree of Norm for triplet loss. One of L1 or L2.')
    # Generator loss
    parser.add_argument('--lambda_enc', type = float, default = 1,
                    help = 'Multipler for structural loss')
    parser.add_argument('--lambda_gan', type = float, default = 1,
                    help = 'Multiplier for discriminator loss for fake image')
    # Discriminator loss
    parser.add_argument('--lambda_d', type = float, default = 1,
                    help = 'Multiplier for discriminator loss for fake image')

    # Training related
    parser.add_argument('--iter_g', type = int, default = 1,
                    help = 'No. of iterations to train generator.')
    parser.add_argument('--iter_d', type = int, default = 1,
                    help = 'No. of iterations to train discriminator.')
    parser.add_argument('--batch_size', type = int, default = 8,
                    help = 'Size of mini batch during training')
    parser.add_argument('--epochs', type = int, default = 300,
                    help = 'No. of epochs of training')
    parser.add_argument('--resume_training', type = str2bool, default = False,
                    help = 'Resume training from latest checkpoint')
    parser.add_argument('--lr_g', type = float, default = 1e-4,
                    help = 'Initial learning rate for generator and encoders')
    parser.add_argument('--lr_d', type = float, default = 1e-4,
                    help = 'Initial learning rate for discriminator')
    parser.add_argument('--train_ratio', type = int, default = None)

    # Validation related
    parser.add_argument('--perform_validation_every', type = int, default = 2500,
                    help = 'Perform validation after how many iterations')
    parser.add_argument('--num_samples', type = int, default = 512,
                    help = 'No. of samples on which validation is to be performed')
    parser.add_argument('--dump_videos', type = str2bool, default = False,
                    help = 'Whether to dump videos during validation')
    parser.add_argument('--dump_path', type = str, default = 'validation_videos',
                    help = 'Directory where videos are to be dumped')

    # Checkpoint related
    parser.add_argument('--id', type = str, required = True,
                    help = 'An ID identifying the training run')
    parser.add_argument('--checkpoint_path', type = str, default = 'checkpoints/',
                    help = 'Path where checkpoints are to be saved')
    parser.add_argument('--save_checkpoint_every', type = int, default = 1000,
                    help = 'Save checkpoint after how many iterations')
    parser.add_argument('--log_losses_every', type = int, default = 10,
                    help = 'Store losses after how many iterations')

    # Dataset related
    parser.add_argument('--train_csv', type = str, default = 'data/train.csv',
                    help = 'Path to the csv file containing information about dataset')
    parser.add_argument('--num_keypoints', type = int, default = 136,
                    help = 'No. of keypoints')
    parser.add_argument('--flatten', type = str2bool, default = False,
                    help = 'Whether to flatten keypoints')
    parser.add_argument('--speaker', type = str, default = 'human',
                    help = '')

    parser.add_argument('--device', type = str, choices = ['cuda', 'cpu'], default = 'cuda',
                    help = 'One of cuda or cpu')

    args = parser.parse_args()

    assert args.log_losses_every % (args.iter_d + args.iter_g) == 0, 'Losses has to be logged after training all the components'
    assert args.save_checkpoint_every % args.log_losses_every == 0, 'Checkpoint has to be saved after logging latest loss'

    return args
