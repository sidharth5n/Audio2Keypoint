import os
import time
import torch
import torch.nn as nn
from torch import optim

from model import Audio2Keypoint
from dataset import DataLoader
from utils import KeyPointsRegLoss, load, save
import eval_utils
from opts import parse_args

def train(args):

    checkpoint_path = os.path.join(args.checkpoint_path, args.id)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    model = Audio2Keypoint(args).to(device)
    model.train()

    Keypts_regloss = KeyPointsRegLoss(args.reg_loss_type, args.d_input, args.lambda_motion_reg_loss)
    Disc_loss = nn.MSELoss()
    Enc_loss = nn.TripletMarginLoss(margin = args.margin, p = args.enc_loss_type)
    G_Enc_loss = nn.L1Loss()

    eval_loss_fns = {'D' : nn.MSELoss(reduction = 'none'),
                     'G' : nn.L1Loss(reduction = 'none'),
                     'K' : KeyPointsRegLoss(args.reg_loss_type, args.d_input, args.lambda_motion_reg_loss, reduction = 'none')}

    D_optim = optim.Adam(model.discriminator.parameters(), lr = args.lr_d)
    E_optim = optim.Adam(model.encoder.parameters(), lr = args.lr_g)
    G_optim = optim.Adam(model.generator.parameters(), lr = args.lr_g)

    if args.resume_training:
        infos = load(os.path.join(checkpoint_path, 'infos.pkl'))
        histories = load(os.path.join(checkpoint_path, 'histories.pkl'))
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'model.pt'), map_location = device))
        D_optim.load_state_dict(torch.load(os.path.join(checkpoint_path, 'D-optimizer.pt'), map_location = device))
        E_optim.load_state_dict(torch.load(os.path.join(checkpoint_path, 'E-optimizer.pt'), map_location = device))
        G_optim.load_state_dict(torch.load(os.path.join(checkpoint_path, 'G-optimizer.pt'), map_location = device))
    else:
        infos, histories = dict(), dict()

    train_loader = DataLoader(args, 'train', infos.get('loader', None))
    val_loader = DataLoader(args, 'dev', length = args.num_samples)

    G_loss_history = histories.get('G_loss_history', {})
    E_loss_history = histories.get('E_loss_history', {})
    D_loss_history = histories.get('D_loss_history', {})

    iteration = infos.get('iter', 0)
    start_epoch = infos.get('epoch', 0)
    count = infos.get('count', 0)
    train_D = infos.get('train_D', True)
    best_loss = infos.get('best_loss', None)
    best_pck = infos.get('pck', None)
    G_loss = G_loss_history.get(iteration, 0)
    E_loss = E_loss_history.get(iteration, 0)
    D_loss = D_loss_history.get(iteration, 0)
    start = time.time()

    for epoch in range(start_epoch, args.epochs):
        for data in train_loader:
            audio_spect, real_pose = [x.to(device) for x in data]

            if train_D:
                # Update D
                real_pose_score, fake_pose_score_detached = model(audio_spect, real_pose, mode = 'train_D')
                # Compute discriminator loss
                D_loss_real = Disc_loss(real_pose_score, torch.ones_like(real_pose_score))
                D_loss_fake = Disc_loss(fake_pose_score_detached, torch.zeros_like(fake_pose_score_detached))
                D_loss = D_loss_real + args.lambda_d * D_loss_fake
                D_optim.zero_grad()
                D_loss.backward()
                D_optim.step()
            else:
                # Update G
                img_enc_piv, fake_pose, real_enc, fake_enc, fake_pose_score = model(audio_spect, real_pose, mode = 'train_G')
                # Loss for training the generator from the global D - have I fooled the global D?
                D_loss_fake = Disc_loss(fake_pose_score, torch.ones_like(fake_pose_score))
                # Regression loss on motion or pose
                pose_regloss = Keypts_regloss(real_pose, fake_pose, model.keypoints, args.train_ratio)
                # Compute generator loss
                G_enc_loss = G_Enc_loss(fake_enc, img_enc_piv)
                G_loss = pose_regloss + (args.lambda_gan * D_loss_fake) + (args.lambda_enc * G_enc_loss)
                G_optim.zero_grad()
                G_loss.backward()
                G_optim.step()

                # Update PIV Encoder
                img_enc_piv, fake_pose, real_enc, fake_enc, fake_pose_score = model(audio_spect, real_pose, mode = 'train_E')
                # Loss for training the generator from the global D - have I fooled the global D?
                D_loss_fake = Disc_loss(fake_pose_score, torch.ones_like(fake_pose_score))
                # Regression loss on motion or pose
                pose_regloss = Keypts_regloss(real_pose, fake_pose, model.keypoints, args.train_ratio)
                # Compute encoder loss
                E_enc_loss = Enc_loss(img_enc_piv, real_enc, fake_enc)
                E_loss = pose_regloss + D_loss_fake + E_enc_loss
                E_optim.zero_grad()
                E_loss.backward()
                E_optim.step()

            end = time.time()

            if iteration % (args.iter_d + args.iter_g) == 0:
                print(f'Iter {iteration} (epoch {epoch}/{args.epochs}) : Disc loss = {D_loss:.2f}, Enc loss = {E_loss:.2f}, Gen loss = {G_loss:.2f}, time/batch = {end-start:.3f}')

            if train_D and (count % args.iter_d == 0):
                train_D = False
                count = 0
            elif not train_D and (count % args.iter_g == 0):
                train_D = True
                count = 0

            count += 1
            iteration += 1

            if (iteration % args.log_losses_every) == 0:
                D_loss_history[iteration] = D_loss.item()
                E_loss_history[iteration] = E_loss.item()
                G_loss_history[iteration] = G_loss.item()

            if (iteration % args.perform_validation_every) == 0:
                model.eval()
                val_loss, pck = eval_utils.eval_split(model, val_loader, eval_loss_fns, args, device, True, [0, 0], [0, 0])
                model.train()
                print(f'Validation loss : {val_loss}, PCK : {pck}')

                if best_loss is None or val_loss < best_loss:
                    best_loss = val_loss
                    best_pck = pck
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_best.pt'))
                    torch.save(D_optim.state_dict(), os.path.join(checkpoint_path, 'D-optimizer_best.pt'))
                    torch.save(E_optim.state_dict(), os.path.join(checkpoint_path, 'E-optimizer_best.pt'))
                    torch.save(G_optim.state_dict(), os.path.join(checkpoint_path, 'G-optimizer_best.pt'))

                    infos = {'iter'      : iteration,
                             'epoch'     : epoch,
                             'count'     : count,
                             'train_D'   : train_D,
                             'best_loss' : best_loss,
                             'pck'       : best_pck,
                             'loader'    : train_loader.state_dict(),
                             'args'      : args}

                    histories = {'D_loss_history' : D_loss_history,
                                 'E_loss_history' : E_loss_history,
                                 'G_loss_history' : G_loss_history}

                    save(infos, os.path.join(checkpoint_path, 'infos_best.pkl'))
                    save(histories, os.path.join(checkpoint_path, 'histories_best.pkl'))
                    print(f'Best checkpoint saved to {checkpoint_path}')

            if (iteration % args.save_checkpoint_every) == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model.pt'))
                torch.save(D_optim.state_dict(), os.path.join(checkpoint_path, 'D-optimizer.pt'))
                torch.save(E_optim.state_dict(), os.path.join(checkpoint_path, 'E-optimizer.pt'))
                torch.save(G_optim.state_dict(), os.path.join(checkpoint_path, 'G-optimizer.pt'))

                infos = {'iter'      : iteration,
                         'epoch'     : epoch,
                         'count'     : count,
                         'train_D'   : train_D,
                         'best_loss' : best_loss,
                         'pck'       : best_pck,
                         'loader'    : train_loader.state_dict(),
                         'args'      : args}

                histories = {'D_loss_history' : D_loss_history,
                             'E_loss_history' : E_loss_history,
                             'G_loss_history' : G_loss_history}

                save(infos, os.path.join(checkpoint_path, 'infos.pkl'))
                save(histories, os.path.join(checkpoint_path, 'histories.pkl'))

                print(f'Checkpoint saved to {checkpoint_path}')

            start = time.time()

if __name__ == '__main__':
    args = parse_args()
    train(args)
