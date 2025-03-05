import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.data_management.prioritized_replay_buffer import PrioritizedReplayBuffer
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.torch.algorithms.adv_irl.utility.bypass_bn import enable_running_stats, disable_running_stats


class AdvIRL(TorchBaseAlgorithm):
    def __init__(
            self,
            mode,  # gail, or ddpm
            discriminator,
            policy_trainer,
            expert_replay_buffer,
            fake_expert_replay_buffer,
            state_only=False,
            disc_optim_batch_size=1024,
            policy_optim_batch_size=1024,
            policy_optim_batch_size_from_expert=0,
            num_update_loops_per_train_call=1,
            num_disc_updates_per_loop_iter=100,
            num_policy_updates_per_loop_iter=100,
            disc_lr=1e-3,
            disc_momentum=0.0,
            disc_optimizer_class=optim.Adam,
            use_grad_pen=True,
            grad_pen_weight=10,
            rew_clip_min=None,
            rew_clip_max=None,
            disc_ddpm=False,
            **kwargs
    ):
        assert mode in [
            "gail",
            "ddpm",
        ], "Invalid adversarial irl algorithm!"
        # if kwargs['wrap_absorbing']: raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.disc_ddpm = disc_ddpm
        self.state_only = state_only

        self.expert_replay_buffer = expert_replay_buffer

        self.fake_expert_replay_buffer = fake_expert_replay_buffer
        if isinstance(self.expert_replay_buffer, list):
            self.replay_mode = "multi_priority"
        elif isinstance(self.expert_replay_buffer, EnvReplayBuffer):
            self.replay_mode = "normal"
        else:
            raise ValueError("Invalid replay buffer mode!")
        self.allow_fake_expert = False
        self.policy_trainer = policy_trainer
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert

        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(), lr=disc_lr, betas=(disc_momentum, 0.999)
        )
        self.disc_expert_optim_batch_size = disc_optim_batch_size // len(expert_replay_buffer) \
            if self.replay_mode == "multi_priority" \
            else disc_optim_batch_size
        self.disc_policy_optim_batch_size = disc_optim_batch_size
        print("\n\nDISC MOMENTUM: %f\n\n" % disc_momentum)

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(disc_optim_batch_size, 1),
                torch.zeros(disc_optim_batch_size, 1),
            ],
            dim=0,
        )
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)

        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.clip_min_rews = rew_clip_min is not None
        self.clip_max_rews = rew_clip_max is not None

        self.disc_eval_statistics = None
        self.threshold = 1

    # get batch from replay buffer
    def get_batch(self, batch_size, from_expert, keys=None):
        if from_expert:
            if self.replay_mode == "multi_priority":
                expert_batches, idxs, is_weights = [], [], []
                for i in range(len(self.expert_replay_buffer)):
                    expert_batch, idx, is_weight = self.expert_replay_buffer[i].random_batch(batch_size, keys)
                    expert_batches.append(expert_batch)
                    idxs.append(idx)
                    is_weights.append(is_weight)
                idx, is_weight = np.concatenate(idxs), np.concatenate(is_weights)
                batch = np_to_pytorch_batch(
                    {key: np.concatenate([batch[key] for batch in expert_batches], axis=0) for key in keys}
                )
                return batch, idx, is_weight
            else:
                batch = self.expert_replay_buffer.random_batch(batch_size, keys)
                batch = np_to_pytorch_batch(batch)
                return batch
        else:
            batch = self.replay_buffer.random_batch(batch_size, keys)
            batch = np_to_pytorch_batch(batch)
            return batch

    def _end_epoch(self):
        self.policy_trainer.end_epoch()
        self.disc_eval_statistics = None
        super()._end_epoch()

    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.disc_eval_statistics)
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
        super().evaluate(epoch)

    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            if "gail" in self.mode in self.mode:
                for _ in range(self.num_disc_updates_per_loop_iter):
                    self._do_reward_training(epoch)
            elif "ddpm" in self.mode:
                for _ in range(self.num_disc_updates_per_loop_iter):
                    self._do_ddpm_reward_training(epoch)
            for _ in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch)

    def _do_ddpm_reward_training(self, epoch):
        """
                Train the discriminator
                """
        params_old = self.discriminator.model.parameters()
        params_old_list = []
        for param in params_old:
            params_old_list.append(param.data)

        keys = ["observations"]
        if self.state_only:
            keys.append("next_observations")
        else:
            keys.append("actions")
        if self.wrap_absorbing:
            keys.append("absorbing")

        if self.replay_mode != "normal":
            expert_batch, idx, is_weight = self.get_batch(self.disc_expert_optim_batch_size, True, keys)
        else:
            expert_batch = self.get_batch(self.disc_expert_optim_batch_size, True, keys)

        if self.fake_expert_replay_buffer.num_steps_can_sample() > 0:
            self.allow_fake_expert = True
            fake_expert_batch, fake_idx, fake_weight = self.fake_expert_replay_buffer.random_batch(
                self.disc_policy_optim_batch_size * 7,
                keys)  # fake expert : expert = 7:1
            fake_expert_batch = np_to_pytorch_batch(fake_expert_batch)
        policy_batch = self.get_batch(self.disc_policy_optim_batch_size, False, keys)

        expert_obs = expert_batch["observations"]
        fake_expert_obs = fake_expert_batch["observations"] if self.allow_fake_expert else None
        policy_obs = policy_batch["observations"]

        if self.wrap_absorbing:
            # pass
            expert_obs = torch.cat(
                [expert_obs, expert_batch["absorbing"][:, 0:1]], dim=-1
            )
            fake_expert_obs = torch.cat(
                [fake_expert_obs, fake_expert_batch["absorbing"][:, 0:1]], dim=-1
            ) if self.allow_fake_expert else None
            policy_obs = torch.cat(
                [policy_obs, policy_batch["absorbing"][:, 0:1]], dim=-1
            )

        if self.state_only:
            expert_next_obs = expert_batch["next_observations"]
            fake_expert_next_obs = fake_expert_batch["next_observations"] if self.allow_fake_expert else None
            policy_next_obs = policy_batch["next_observations"]
            if self.wrap_absorbing:
                expert_next_obs = torch.cat(
                    [expert_next_obs, expert_batch["absorbing"][:, 1:]], dim=-1
                )
                fake_expert_obs = torch.cat(
                    [fake_expert_next_obs, fake_expert_batch["absorbing"][:, 1:]], dim=-1
                ) if self.allow_fake_expert else None
                policy_next_obs = torch.cat(
                    [policy_next_obs, policy_batch["absorbing"][:, 1:]], dim=-1
                )
            expert_disc_input = torch.cat([expert_obs, expert_next_obs], dim=1)
            fake_expert_disc_input = torch.cat([fake_expert_obs, fake_expert_next_obs],
                                               dim=1) if self.allow_fake_expert else None
            policy_disc_input = torch.cat([policy_obs, policy_next_obs], dim=1)
        else:
            expert_acts = expert_batch["actions"]
            fake_expert_acts = fake_expert_batch["actions"] if self.allow_fake_expert else None
            policy_acts = policy_batch["actions"]
            expert_disc_input = torch.cat([expert_acts, expert_obs], dim=1)
            fake_expert_disc_input = torch.cat([fake_expert_acts, fake_expert_obs],
                                               dim=1) if self.allow_fake_expert else None
            policy_disc_input = torch.cat([policy_acts, policy_obs], dim=1)


        expert_d = self.discriminator.diffusion.loss(expert_disc_input, disc_ddpm=self.disc_ddpm).unsqueeze(
            dim=1)
        fake_expert_d = self.discriminator.diffusion.loss(fake_expert_disc_input, disc_ddpm=self.disc_ddpm).unsqueeze(
            dim=1) if self.allow_fake_expert else None
        actor_d = self.discriminator.diffusion.loss(policy_disc_input, disc_ddpm=self.disc_ddpm).unsqueeze(dim=1)
        if self.replay_mode != "normal":
            bce = torch.nn.BCELoss(reduction='none')
            w = torch.tensor(is_weight, device=ptu.device, dtype=torch.float32)
            expert_loss = (w * bce(expert_d, torch.ones(expert_d.size(), device=ptu.device))).mean()
        else:
            expert_loss = torch.nn.BCELoss()(expert_d, torch.ones(expert_d.size(), device=ptu.device))  # target=1
        actor_loss = torch.nn.BCELoss()(actor_d, torch.zeros(actor_d.size(), device=ptu.device))  # target=0
        if self.allow_fake_expert:
            fake_bce = torch.nn.BCELoss(reduction='none')
            fake_w = torch.tensor(fake_weight, device=ptu.device, dtype=torch.float32)
            fake_expert_loss = (fake_w * fake_bce(fake_expert_d, torch.ones(fake_expert_d.size(),
                                                                            device=ptu.device))).mean()
        else:
            fake_expert_loss = 0.0

        loss = expert_loss + actor_loss + fake_expert_loss

        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)
            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)

            out = self.discriminator.diffusion.loss(interp_obs, disc_ddpm=self.disc_ddpm).sum()

            gradients = autograd.grad(
                outputs=out,
                inputs=interp_obs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradient_penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        total_loss = loss + disc_grad_pen_loss

        self.discriminator.diff_opti.zero_grad()
        total_loss.backward()
        self.discriminator.diff_opti.step()

        abs_errors = torch.abs(
            expert_d - torch.ones(expert_d.size(), device=ptu.device)).detach().cpu().numpy().squeeze()
        if self.replay_mode == "multi_priority":
            for i in range(len(self.expert_replay_buffer)):
                size = self.disc_expert_optim_batch_size
                start = i * size
                end = (i + 1) * size
                self.expert_replay_buffer[i].update(idx[start:end], abs_errors[start:end])
        if self.allow_fake_expert:
            fake_abs_errors = torch.abs(
                fake_expert_d - torch.ones(fake_expert_d.size(), device=ptu.device)).detach().cpu().numpy().squeeze()
            self.fake_expert_replay_buffer.update(fake_idx, fake_abs_errors)

        # update threshold
        self.discriminator.diffusion.eval()
        disc_logits = self.discriminator.disc_reward(expert_disc_input)
        self.threshold = disc_logits.mean().item()
        self.discriminator.diffusion.train()
        """
        Save some statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            self.disc_eval_statistics["Disc ddpm Loss"] = np.mean(
                ptu.get_numpy(loss)
            )

            self.disc_eval_statistics["Disc ddpm expert Loss"] = np.mean(
                ptu.get_numpy(expert_loss)
            )

            self.disc_eval_statistics["Disc ddpm actor Loss"] = np.mean(
                ptu.get_numpy(actor_loss)
            )
            if self.use_grad_pen:
                self.disc_eval_statistics["Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.disc_eval_statistics["Grad Pen W"] = np.mean(self.grad_pen_weight)

            # log lr
            self.disc_eval_statistics["Disc lr"] = self.discriminator.diff_opti.param_groups[0]['lr']

    def _do_reward_training(self, epoch):
        """
        Train the discriminator
        """
        self.disc_optimizer.zero_grad()

        keys = ["observations"]
        if self.state_only:
            keys.append("next_observations")
        else:
            keys.append("actions")
        if self.wrap_absorbing:
            keys.append("absorbing")

        if self.replay_mode == "multi_priority":
            expert_batch, idx, is_weight = self.get_batch(self.disc_expert_optim_batch_size, True, keys)
        else:
            expert_batch = self.get_batch(self.disc_expert_optim_batch_size, True, keys)
        policy_batch = self.get_batch(self.disc_policy_optim_batch_size, False, keys)

        expert_obs = expert_batch["observations"]
        policy_obs = policy_batch["observations"]

        if self.wrap_absorbing:
            # pass
            expert_obs = torch.cat(
                [expert_obs, expert_batch["absorbing"][:, 0:1]], dim=-1
            )
            policy_obs = torch.cat(
                [policy_obs, policy_batch["absorbing"][:, 0:1]], dim=-1
            )

        if self.state_only:
            expert_next_obs = expert_batch["next_observations"]
            policy_next_obs = policy_batch["next_observations"]
            if self.wrap_absorbing:
                # pass
                expert_next_obs = torch.cat(
                    [expert_next_obs, expert_batch["absorbing"][:, 1:]], dim=-1
                )
                policy_next_obs = torch.cat(
                    [policy_next_obs, policy_batch["absorbing"][:, 1:]], dim=-1
                )
            expert_disc_input = torch.cat([expert_obs, expert_next_obs], dim=1)
            policy_disc_input = torch.cat([policy_obs, policy_next_obs], dim=1)
        else:
            expert_acts = expert_batch["actions"]
            policy_acts = policy_batch["actions"]
            expert_disc_input = torch.cat([expert_obs, expert_acts], dim=1)
            policy_disc_input = torch.cat([policy_obs, policy_acts], dim=1)
        disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)
        disc_logits = self.discriminator(disc_input)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)

            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)
            a = self.discriminator(interp_obs).sum()
            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs).sum(),
                inputs=[interp_obs],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()
        self.disc_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            self.disc_eval_statistics["Disc CE Loss"] = np.mean(
                ptu.get_numpy(disc_ce_loss)
            )
            self.disc_eval_statistics["Disc Acc"] = np.mean(ptu.get_numpy(accuracy))
            if self.use_grad_pen:
                self.disc_eval_statistics["Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.disc_eval_statistics["Grad Pen W"] = np.mean(self.grad_pen_weight)

    def _do_policy_training(self, epoch):
        if self.policy_optim_batch_size_from_expert > 0:
            policy_batch_from_policy_buffer = self.get_batch(
                self.policy_optim_batch_size - self.policy_optim_batch_size_from_expert,
                False,
            )

            policy_batch_from_expert_buffer = self.get_batch(
                self.policy_optim_batch_size_from_expert, True
            )
            policy_batch = {}
            for k in policy_batch_from_policy_buffer:
                policy_batch[k] = torch.cat(
                    [
                        policy_batch_from_policy_buffer[k],
                        policy_batch_from_expert_buffer[k],
                    ],
                    dim=0,
                )
        else:
            policy_batch = self.get_batch(self.policy_optim_batch_size, False)

        obs = policy_batch["observations"]
        acts = policy_batch["actions"]
        next_obs = policy_batch["next_observations"]
        if self.wrap_absorbing:
            # pass
            obs = torch.cat([obs, policy_batch["absorbing"][:, 0:1]], dim=-1)
            next_obs = torch.cat([next_obs, policy_batch["absorbing"][:, 1:]], dim=-1)

        if "ddpm" in self.mode:
            self.discriminator.eval()
            if self.state_only:
                disc_input = torch.cat([obs, next_obs], dim=1)
            else:
                disc_input = torch.cat([acts, obs], dim=1)

            self.discriminator.diffusion.eval()
            disc_logits, fake_expert_simple = self.discriminator.disc_reward(
                disc_input, simple=True, threshold=self.threshold)
            disc_logits, fake_expert_simple = disc_logits.detach(), fake_expert_simple.detach().cpu().numpy()
            # add sample to fake expert replay buffer
            for out in fake_expert_simple:
                if self.state_only:
                    fake_expert_obs = out[:obs.shape[1]]
                    fake_expert_next_obs = out[obs.shape[1]:]
                    if self.wrap_absorbing:
                        fake_expert_obs = fake_expert_obs[:-1]
                        fake_expert_next_obs = fake_expert_next_obs[:-1]
                else:
                    fake_expert_acts = out[:acts.shape[1]]
                    fake_expert_obs = out[acts.shape[1]:]
                    if self.wrap_absorbing:
                        fake_expert_obs = fake_expert_obs[:-1]
                self.fake_expert_replay_buffer.add_sample(
                    observation=fake_expert_obs,
                    action=None if self.state_only else fake_expert_acts,
                    reward=None,
                    terminal=np.array([False]),
                    next_observation=fake_expert_next_obs if self.state_only else None
                )

            self.discriminator.diffusion.train()
        else:
            self.discriminator.eval()
            if self.state_only:
                disc_input = torch.cat([obs, next_obs], dim=1)
            else:
                disc_input = torch.cat([obs, acts], dim=1)

            disc_logits = self.discriminator(disc_input).detach()

            self.discriminator.train()
        # compute the reward using the algorithm
        if self.mode == "gail":
            policy_batch["rewards"] = F.softplus(
                disc_logits, beta=-1
            )
        elif self.mode == "ddpm":
            d = disc_logits.to(ptu.device)
            policy_batch["rewards"] = (- torch.log(1 - d)).unsqueeze(dim=1)

        if self.clip_max_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], max=self.rew_clip_max
            )
        if self.clip_min_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], min=self.rew_clip_min
            )

        # policy optimization step
        self.policy_trainer.train_step(policy_batch)

        # logs
        self.disc_eval_statistics["Disc Rew Mean"] = np.mean(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Std"] = np.std(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Max"] = np.max(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Min"] = np.min(
            ptu.get_numpy(policy_batch["rewards"])
        )

    @property
    def networks(self):
        return [self.discriminator] + self.policy_trainer.networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(disc=self.discriminator)
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot

    def to(self, device):
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)
        super().to(device)
