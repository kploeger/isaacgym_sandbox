"""
    mail@kaiploeger.net
"""

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3)
import time

from policies import CubicMP, PiecewiseMP
from envs import IsaacSim, IsaacJugglingWam4

from ereps import eREPS

N_ENVS = 24

# --- new ---
# # stroke based movement primitive to get going
# poss_stroke = np.array([[- 0.10, 1.20, 0.0, 1.25],
                        # [- 0.09, 1.01, 0.0, 0.99],
                        # [+ 0.08, 1.16, 0.0, 1.19],
                        # [+ 0.112, 0.95, 0.0, 1.09],
                        # [- 0.08, 1.18, 0.0, 1.22]])
# vels_stroke = np.zeros_like(poss_stroke)
# times_stroke = np.array([0.0, 0.095, 0.475, 0.57, 0.95])


# # cyclic movement primitive
# poss_cyclic = np.array([[- 0.08, 1.19, 0.0, 1.22],
                        # [- 0.12, 1.03, 0.0, 0.95],
                        # [+ 0.08, 1.19, 0.0, 1.22],
                        # [+ 0.12, 1.03, 0.0, 0.95],
                        # [- 0.08, 1.19, 0.0, 1.22]])
# vels_cyclic = np.zeros_like(poss_cyclic)
# times_cyclic = np.array([0.0, 0.095, 0.475, 0.57, 0.95])

# open_params = ['s_q1_t1', 's_q2_t1', 's_q4_t1',
               # 's_q1_t2', 's_q2_t2', 's_q4_t2',
               # 's_q1_t3', 's_q2_t3', 's_q4_t3',
               # 's_t1', 's_t2', 's_t3',
               # 'c_q1_t0', 'c_q2_t0', 'c_q4_t0',
               # 'c_q1_t1', 'c_q2_t1', 'c_q4_t1',
               # 'c_q1_t2',
               # 'c_q1_t3',
               # 'c_t1']

# constraints = {'equal':  [['s_q1_t4', 'c_q1_t0'], # smooth transition stroke -> cyclic
                          # ['s_q2_t4', 'c_q2_t0'],
                          # ['s_q4_t4', 'c_q4_t0'],
                          # ['c_q2_t2', 'c_q2_t0'], # q2 and q3 move symmetrical in cyclic
                          # ['c_q4_t2', 'c_q4_t0'],
                          # ['c_q2_t3', 'c_q2_t1'],
                          # ['c_q4_t3', 'c_q4_t1'],
                          # ['c_q1_t4', 'c_q1_t0'], # smooth transition cyclic -> cyclic
                          # ['c_q2_t4', 'c_q2_t0'],
                          # ['c_q4_t4', 'c_q4_t0']],

               # 'offset': [['c_t3',         'c_t1',       0.475]],
               # 'mirror': []}


# --- old ---

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

        print('Func: start eveal')
        for i in range(self.nb_envs):
            self.policies[i].set_params(params_batch[i])
            q0, _, _ = self.policies[i].get_action(0)
            self.sim.reset(q=q0)

            # trafo = sim.envs[0].get_ee_trafo()
            # print(trafo.p)
        print('Func: done resetting')

        kt = 0
        returns = np.zeros(self.nb_envs)
        for _ in range(1000):

            # if kt <= 1:
                # for k_envs in range(self.nb_envs):
                    # bpos = self.sim.envs[k_envs].get_ball_positions()
                    # x = self.sim.envs[k_envs].get_ee_trafo().p
                    # p1 = bpos[0]
                    # p2 = bpos[1]
                    # print(kt, k_envs, p1, x)
            for k_envs in range(self.nb_envs):
                q_des, dq_des, tau_des = self.policies[k_envs].get_action(kt*self.dt)
                self.sim.envs[k_envs].apply_action(q_des, dq_des, tau_des)
                # poss = self.sim.envs[0].get_ball_positions()
                # print(poss[0])
                # print(poss[1])
                # time.sleep(3)
            kt = self.sim.step()
            # print(kt, kt*self.dt, self.sim.gym.get_sim_time(self.sim.sim), self.sim.envs[0].motor_torques, self.sim.envs[0].vel)
            # print(self.sim.gym.get_sim_time(self.sim.sim), q_des, dq_des)

            q.append(self.sim.envs[0].pos)
            dq.append(self.sim.envs[0].vel)
            returns += reward(self.sim)

        # q = np.array(q)
        # dq = np.array(dq)

        # plt.figure()
        # plt.plot(q[:, 1])
        # plt.show()

        print(np.sort(returns))


        return returns


def main():
    nb_envs = 7

    policies = []
    for _ in range(nb_envs):
        policy_stroke = CubicMP(poss_stroke, vels_stroke, times_stroke, cyclic=False, id='s')
        policy_cyclic = CubicMP(poss_cyclic, vels_cyclic, times_cyclic, cyclic=True,  id='c')
        intervals = np.array([policy_stroke.duration, np.inf])
        policy = PiecewiseMP([policy_stroke, policy_cyclic], intervals, visible_params=param_names, constraints=constraints)
        policies.append(policy)

    # # some checks:
    # all_param_names = policies[0].list_all_param_names()
    open_param_names = policies[0].get_open_param_names()
    print(open_param_names)
    # param_mask = policies[0].param_mask
    # dm_policy = sum(param_mask)
    # dependent_param_names = policies[0].get_constrained_param_names()
    # print(list(set(open_param_names) & set(dependent_param_names)))
    # const_param_names = list(set(all_param_names) - set(open_param_names) - set(dependent_param_names))
    # const_param_names.sort()
    # print(const_param_names)
    # print(f'{len(all_param_names)} parameters')
    # print(f'{len(open_param_names)} open parameters')
    # print(f'{len(dependent_param_names)} dependent parameters')
    # print(f'{len(const_param_names)} constant hidden parameters')

    sim = IsaacSim(IsaacJugglingWam4, num_envs=nb_envs)

    print('juggle: sim created')

    mu0 = policies[0].get_params()

    func = Func(sim, policies)

    # reps = eREPS(func=func, n_episodes=nb_envs, kl_bound=20, mu0=mu0, cov0=5e-4)
    # reps = eREPS(func=func, n_episodes=nb_envs, kl_bound=20, mu0=mu0, cov0=0.0025)

    params = policy.get_params()
    # params_vec = np.tile(params, (nb_envs, 1))
    params_vec = np.tile(mu0, (nb_envs, 1))

    print(np.shape(params_vec))


    func.eval(params_vec)

    # reps.run(10, verbose=True)


if __name__ == '__main__':
    main()


