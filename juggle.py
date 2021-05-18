"""
    mail@kaiploeger.net
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from policies import CubicMP, PiecewiseMP
from envs import IsaacSim, IsaacJugglingWam4
from ereps import eREPS


# stroke based movement primitive to get going
poss_stroke = np.array([[- 0.10, 1.20, 0.0, 1.25],
                        [- 0.09, 1.01, 0.0, 0.99],
                        [+ 0.08, 1.16, 0.0, 1.19],
                        [+ 0.112, 0.95, 0.0, 1.09],
                        [- 0.08, 1.18, 0.0, 1.22]])
vels_stroke = np.zeros_like(poss_stroke)
times_stroke = np.array([0.095, 0.475, 0.57, 0.95])


# cyclic movement primitive
poss_cyclic = np.array([[- 0.08, 1.19, 0.0, 1.22],
                        [- 0.12, 1.03, 0.0, 0.95],
                        [+ 0.08, 1.19, 0.0, 1.22],
                        [+ 0.12, 1.03, 0.0, 0.95]])
vels_cyclic = np.zeros_like(poss_cyclic)
times_cyclic = np.array([0.095, 0.475, 0.57, 0.95])


# visible parameters to be trained
param_names = ['pos1_act0_s', 'pos1_act1_s', 'pos1_act3_s',
               'pos2_act0_s', 'pos2_act1_s', 'pos2_act3_s',
               'pos3_act0_s', 'pos3_act1_s', 'pos3_act3_s',
               't1_s', 't2_s', 't3_s',
               'pos0_act0_c', 'pos0_act1_c', 'pos0_act3_c',
               'pos1_act0_c', 'pos1_act1_c', 'pos1_act3_c',
               'pos2_act0_c',
               'pos3_act0_c',
               't1_c']

# cosntraints on hidden parameters
constraints = {'equal':  [['pos4_act0_s', 'pos0_act0_c'],
                          ['pos4_act1_s', 'pos0_act1_c'],
                          ['pos4_act3_s', 'pos0_act3_c'],
                          ['pos2_act1_c', 'pos0_act1_c'],
                          ['pos2_act3_c', 'pos0_act3_c'],
                          ['pos3_act1_c', 'pos1_act1_c'],
                          ['pos3_act3_c', 'pos1_act3_c']],
               'offset': [['t3_c',         't1_c',       0.475]],
               'mirror': []}


def reward(sim):
    rewards = np.zeros(len(sim.envs))
    for i in range(len(sim.envs)):
        env = sim.envs[i]
        x_balls = env.get_ball_positions()
        heigt_balls = np.array([x.z for x in x_balls])
        if min(heigt_balls) > 0.5:
            rewards[i] = 1
    return rewards



class Func:
    """ interface for eREPS to evaluate a batch of parameters """
    def __init__(self, sim, policies):
        self.sim = sim
        self.policies = policies
        self.dm_act = len(policies[0].get_params())
        self.nb_envs = len(self.sim.envs)
        self.dt = self.sim.envs[0].dt


    def eval(self, params_batch):
        q = []
        dq = []
        tau = []

        for i in range(self.nb_envs):
            self.policies[i].set_params(params_batch[i])
            q0, _, _ = self.policies[i].get_action(0)
            self.sim.reset(q=q0)

        kt = 0
        returns = np.zeros(self.nb_envs)
        for _ in range(1000):

            for k_envs in range(self.nb_envs):
                q_des, dq_des, tau_des = self.policies[k_envs].get_action(kt*self.dt)
                self.sim.envs[k_envs].apply_action(q_des, dq_des, tau_des)

            kt = self.sim.step()
            q.append(self.sim.envs[0].pos)
            dq.append(self.sim.envs[0].vel)
            returns += reward(self.sim)

        return returns


def main():
    nb_envs = 30

    policies = []
    for _ in range(nb_envs):
        policy_stroke = CubicMP(poss_stroke, vels_stroke, times_stroke, cyclic=False, id='s')
        policy_cyclic = CubicMP(poss_cyclic, vels_cyclic, times_cyclic, cyclic=True,  id='c')
        intervals = np.array([policy_stroke.duration, np.inf])
        policy = PiecewiseMP([policy_stroke, policy_cyclic], intervals, visible_params=param_names, constraints=constraints)
        policies.append(policy)

    sim = IsaacSim(IsaacJugglingWam4, num_envs=nb_envs)
    func = Func(sim, policies)

    mu0 = policies[0].get_params()
    # reps = eREPS(func=func, n_episodes=nb_envs, kl_bound=20, mu0=mu0, cov0=5e-4)

    # reps.run(10, verbose=True)
    func.eval(np.tile(mu0, (nb_envs, 1)))


if __name__ == '__main__':
    main()


