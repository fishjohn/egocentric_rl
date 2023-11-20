import onnx
import os, copy

import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import Actor, get_activation
from rsl_rl.modules.estimator import Estimator
from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
import argparse

from legged_gym import LEGGED_GYM_ROOT_DIR, WANDB_SAVE_DIR


def export_hardware_vision_as_onnx(policy, path, dummy_input):
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, "policy.onnx")
    model = copy.deepcopy(policy).to()
    model.eval()

    input_names = ["observation"]
    output_names = ["action"]

    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    check_model = onnx.load(model_path)
    onnx.checker.check_model(check_model)

    print("Exported policy as onnx script to: ", model_path)


def export_depth_encoder_as_onnx(depth_encoder, path, dummy_input):
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, "depth_encoder.onnx")
    model = copy.deepcopy(depth_encoder).to()
    model.eval()

    input_names = ["combined_image_proprio"]
    output_names = ["depth_latent"]

    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    check_model = onnx.load(model_path)
    onnx.checker.check_model(check_model)

    print("Exported depth encoder as onnx script to: ", model_path)


def get_load_path(root, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(root, model)
    return load_path, checkpoint


class HardwareVisionNN(nn.Module):
    def __init__(self, num_prop,
                 num_scan,
                 num_priv_latent,
                 num_priv_explicit,
                 num_hist,
                 num_actions,
                 tanh,
                 actor_hidden_dims=[512, 256, 128],
                 scan_encoder_dims=[128, 64, 32],
                 depth_encoder_hidden_dim=512,
                 activation='elu',
                 priv_encoder_dims=[64, 20]
                 ):
        super(HardwareVisionNN, self).__init__()

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        num_obs = num_prop + num_scan + num_hist * num_prop + num_priv_latent + num_priv_explicit
        self.num_obs = num_obs
        activation = get_activation(activation)

        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims,
                           num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=tanh)

        self.estimator = Estimator(input_dim=num_prop, output_dim=num_priv_explicit, hidden_dims=[128, 64])

    def forward(self, obs):
        depth_latent = obs[-32:]
        obs = obs[:self.num_obs]
        obs = obs.reshape(1, -1)
        depth_latent = depth_latent.reshape(1, -1)
        # obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_explicit] = self.estimator(
        #     obs[:, :self.num_prop])
        return self.actor(obs, hist_encoding=True, eval=False, scandots_latent=depth_latent).squeeze()
        # return obs, depth_latent


class HardwareDepthEncoder(nn.Module):
    def __init__(self, num_prop,
                 scandots_output_dim=32,
                 hidden_state_dim=512,
                 device=torch.device('cpu')
                 ):
        super(HardwareDepthEncoder, self).__init__()

        self.num_prop = num_prop

        self.backbone = DepthOnlyFCBackbone58x87(num_prop, scandots_output_dim, hidden_state_dim)

        self.depth_encoder = RecurrentDepthBackbone(self.backbone, num_prop, scandots_output_dim).to(device)

    def forward(self, combined_image_proprio):
        depth_image = combined_image_proprio[:58 * 87].view(1, 58, 87)
        proprioception = combined_image_proprio[58 * 87:].unsqueeze(0)

        return self.depth_encoder(depth_image, proprioception).squeeze()


def play(args):
    load_run = WANDB_SAVE_DIR + "/logs/egocentric_rl/" + args.exptid
    checkpoint = args.checkpoint

    num_actions = 6
    num_scan = 121
    n_priv_explicit = 3 + 3 + 3
    n_priv_latent = 1 + 3 + 1 + 6 + 6
    n_proprio = 3 + 3 + 3 + 6 + 6 + 6 + 2
    history_len = 10

    device = torch.device('cpu')
    policy = HardwareVisionNN(n_proprio, num_scan, n_priv_latent, n_priv_explicit, history_len, num_actions,
                              args.tanh).to(device)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
    load_run = os.path.dirname(load_path)
    print(f"Loading model from: {load_path}")
    ac_state_dict = torch.load(load_path, map_location=device)
    # policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    policy.actor.load_state_dict(ac_state_dict['depth_actor_state_dict'], strict=True)
    policy.estimator.load_state_dict(ac_state_dict['estimator_state_dict'])
    policy = policy.to(device)  # .cpu()

    depth_encoder = HardwareDepthEncoder(n_proprio, scandots_output_dim=32, hidden_state_dim=512, device=device)
    depth_encoder.depth_encoder.load_state_dict(ac_state_dict['depth_encoder_state_dict'])

    if not os.path.exists(os.path.join(load_run, "traced")):
        os.mkdir(os.path.join(load_run, "traced"))
    state_dict = {'depth_encoder_state_dict': ac_state_dict['depth_encoder_state_dict']}
    torch.save(state_dict, os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-vision_weight.pt"))

    # Save the traced actor
    policy.eval()
    with torch.no_grad():
        obs_input = torch.ones(n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len * n_proprio,
                               device=device)
        depth_latent = torch.ones(32, device=device)
        test = policy(torch.cat([obs_input, depth_latent], dim=-1))
        onnx_path = os.path.join(load_run, "exported", args.exptid + "-" + str(checkpoint))
        export_hardware_vision_as_onnx(policy, onnx_path, torch.cat([obs_input, depth_latent], dim=-1))
        depth_image = torch.ones(58, 87, device=device)
        obs_proprio = torch.ones(n_proprio, device=device)
        test = depth_encoder(torch.cat((depth_image.view(-1), obs_proprio), dim=0))
        export_depth_encoder_as_onnx(depth_encoder, onnx_path, torch.cat((depth_image.view(-1), obs_proprio), dim=0))

        traced_policy = torch.jit.trace(policy, torch.cat([obs_input, depth_latent], dim=-1))
        # traced_policy = torch.jit.script(policy)
        save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-base_jit.pt")
        traced_policy.save(save_path)
        print("Saved traced_actor at ", os.path.abspath(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--tanh', action='store_true')
    args = parser.parse_args()
    play(args)
