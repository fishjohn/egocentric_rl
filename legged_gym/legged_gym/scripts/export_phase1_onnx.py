import onnx
import os, sys, copy

import torch
import torch.nn as nn
import argparse

from legged_gym import WANDB_SAVE_DIR
from rsl_rl.modules.actor_critic import Actor, ActorCriticRMA, get_activation
from rsl_rl.modules.estimator import Estimator


class ScandotActorNN(nn.Module):
    def __init__(self,
                 num_prop,
                 num_scan,
                 num_priv_latent,
                 num_priv_explicit,
                 num_hist,
                 num_actions,
                 tanh,
                 actor_hidden_dims=[512, 256, 128],
                 scan_encoder_dims=[128, 64, 32],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 priv_encoder_dims=[64, 20]
                 ):
        super(ScandotActorNN, self).__init__()

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.num_obs = num_prop + num_scan + num_hist * num_prop + num_priv_latent + num_priv_explicit
        self.activation = get_activation(activation)

        policy_cfg = {}
        policy_cfg['priv_encoder_dims'] = [64, 20]
        policy_cfg['tanh_encoder_output'] = tanh
        self.ac = ActorCriticRMA(num_prop, num_scan, self.num_obs, num_priv_latent, num_priv_explicit, num_hist,
                                 num_actions,
                                 scan_encoder_dims, actor_hidden_dims, critic_hidden_dims, **policy_cfg)

        self.actor = self.ac.actor
        self.estimator = Estimator(input_dim=num_prop, output_dim=num_priv_explicit, hidden_dims=[128, 64])

    def forward(self, obs):
        obs = obs.reshape(1, -1)
        # obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_explicit] = self.estimator(
        #     obs[:, :self.num_prop])
        return self.actor(obs, hist_encoding=True, eval=False).squeeze()


def export_scandot_actor_as_onnx(policy, path, dummy_input):
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


def play(args):
    load_run = WANDB_SAVE_DIR + "/logs/egocentric_rl/" + args.exptid
    checkpoint = args.checkpoint

    num_actions = 6
    num_scan = 165
    n_priv_explicit = 3 + 3 + 3
    n_priv_latent = 1 + 3 + 1 + 6 + 6
    n_proprio = 3 + 3 + 3 + 6 + 6 + 6 + 2
    history_len = 10

    device = torch.device('cpu')
    policy = ScandotActorNN(n_proprio, num_scan, n_priv_latent, n_priv_explicit, history_len, num_actions,
                            args.tanh).to(device)
    load_path, checkpoint = get_load_path(load_run, checkpoint)
    load_run = os.path.dirname(load_path)
    print(f"Loading model from: {load_path}")
    ac_state_dict = torch.load(load_path, map_location=device)
    policy.ac.load_state_dict(ac_state_dict['model_state_dict'], strict=True)
    policy.estimator.load_state_dict(ac_state_dict['estimator_state_dict'])
    policy = policy.to(device)

    policy.eval()
    with torch.no_grad():
        obs_input = torch.ones(n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len * n_proprio,
                               device=device)
        test = policy(obs_input)
        onnx_path = os.path.join(load_run, "exported", args.exptid + "-" + str(checkpoint))
        export_scandot_actor_as_onnx(policy, onnx_path, obs_input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--tanh', action='store_true')
    args = parser.parse_args()
    play(args)
