
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Policy:

    def __init__(self):
        pass

    def list_all_param_names(self):
        raise NotImplementedError("{}.list_all_param_names() is not implemented.".format(str(type(self))[8:-2]))

    def get_params(self, param_names=None):
        raise NotImplementedError("{}.get_params() is not implemented.".format(str(type(self))[8:-2]))


class MovementPrimitive:

    def __init__(self, interval, cyclic=False):
        pass

    def reset(self):
        pass

    def get_action(self, time):
        raise NotImplementedError("{}.get_des_state(time) is not implemented.".format(str(type(self))[8:-2]))

    def list_all_param_names(self):
        raise NotImplementedError("{}.list_all_param_names() is not implemented.".format(str(type(self))[8:-2]))

    def get_params(self):
        raise NotImplementedError("{}.get_params() is not implemented.".format(str(type(self))[8:-2]))

    def set_params(self, params):
        raise NotImplementedError("{}.set_params(params) is not implemented.".format(str(type(self))[8:-2]))

    def set_constraints(self, constraints):
        raise NotImplementedError("{}.set_constraints(constraints) is not implemented.".format(str(type(self))[8:-2]))

    def get_mask(self, param_names):
        raise NotImplementedError("{}.get_mask(param_names) is not implemented.".format(str(type(self))[8:-2]))

    def set_active_params(self, param_names):
        raise NotImplementedError("{}.set_active_params(param_names) is not implemented.".format(str(type(self))[8:-2]))

    def plot(self):
        raise NotImplementedError("{}.plot() is not implemented.".format(str(type(self))[8:-2]))


class CubicMP(Policy, MovementPrimitive):

    def __init__(self, pos, vel, times, cyclic=False, visible_params=None, id=None):

        self.pos = pos
        self.vel = vel
        self.times = times
        self.cyclic = cyclic
        self.cycle_duration = self.times[-1]
        if self.cyclic:
            self.duration = np.inf
        else:
            self.duration = self.times[-1]
        if id is None:
            self.id = None
        else:
            self.id = str(id)

        self.times = np.concatenate(([0], self.times))
        if self.cyclic:
            self.pos  = np.concatenate((self.pos, [self.pos[0, :]]), axis=0)
            self.vel = np.concatenate((self.vel, [self.vel[0, :]]), axis=0)

        self.dm_act = self.pos.shape[1]

        for i in range(self.dm_act):
            for j in range(len(self.times)):
                if np.isnan(self.vel[j, i]):
                    jm1 = (j-1)%len(self.times)
                    jp1 = (j+1)%len(self.times)
                    self.vel[j, i] = (self.pos[jp1, i] - self.pos[jm1, i]) / (self.times[jp1] - self.times[jm1])

        self.n_splines = len(self.times) - 1
        self.spline_parameters = np.zeros((self.n_splines, 4, self.dm_act))

        self.fit_splines()

        if visible_params is None:
            self.active_params = self.list_all_param_names()
        else:
            self.active_params = visible_params
        self.param_mask = self.get_mask(self.active_params)
        self.n_params = len(self.active_params)

        MovementPrimitive.__init__(self, self.times[-1], cyclic=cyclic)
        Policy.__init__(self)

    def reset(self):
        pass

    def get_action(self, time):
        if self.cyclic:
            time = time % self.times[-1]

        else:
            time = np.clip(time, self.times[0], self.times[-1])

        # which spline?
        for i in range(self.n_splines):
            if self.times[i] <= time and time <= self.times[i + 1]:
                time = time - self.times[i]

                pos = self.spline_parameters[i, 3, :] * time ** 3 + \
                      self.spline_parameters[i, 2, :] * time ** 2 + \
                      self.spline_parameters[i, 1, :] * time + \
                      self.spline_parameters[i, 0, :]

                vel = 3 * self.spline_parameters[i, 3, :] * time ** 2 + \
                      2 * self.spline_parameters[i, 2, :] * time + \
                      self.spline_parameters[i, 1, :]

                tau = np.zeros_like(pos)

                return pos, vel, tau

    def plot(self, actuators=None, plot_pos=True, plot_vel=False, plot_tau=False):
        if actuators is None:
            actuators = range(self.dm_act)

        t = np.linspace(self.times[0], self.times[-1], 1000)


        pos = np.zeros((len(t), self.dm_act))
        vel = np.zeros((len(t), self.dm_act))
        tau = np.zeros((len(t), self.dm_act))

        for i in range(len(t)):
            pos[i], vel[i], tau[i] = self.get_action(t[i])

        if plot_pos:
            fig_pos = plt.figure()
            for i in actuators:
                ax = fig_pos.add_subplot(len(actuators), 1, i + 1)
                ax.plot(t, pos[:, i])
                if not self.pos is None:
                    ax.scatter(self.times, self.pos[:, i])
                ax.set_ylabel('pos ' + str(i + 1))

        if plot_vel:
            fig_vel = plt.figure()
            for i in actuators:
                ax = fig_vel.add_subplot(len(actuators), 1, i + 1)
                ax.plot(t, vel[:, i])
                if not self.vel is None:
                    ax.scatter(self.times, self.vel[:, i])

                ax.set_ylabel('vel ' + str(i + 1))

        if plot_tau:
            fig_tau = plt.figure()
            for i in actuators:
                ax = fig_tau.add_subplot(len(actuators), 1, i + 1)
                ax.plot(t, tau[:, i])
                ax.set_ylabel('tau ' + str(i + 1))

        plt.show()

    def list_all_param_names(self):
        param_names = []

        # this is for cyclic!!
        if self.cyclic:
            if self.id is None:
                for pos in range(self.pos.shape[0] - 1):
                    for act in range(self.dm_act):
                        param_names.append('pos{}_act{}'.format(pos, act))

                for vel in range(self.vel.shape[0] - 1):
                    for act in range(self.dm_act):
                        param_names.append('vel{}_act{}'.format(vel, act))

                for t in range(1, self.times.shape[0]):
                    param_names.append('t{}'.format(t))
            else:
                for pos in range(self.pos.shape[0] - 1):
                    for act in range(self.dm_act):
                        param_names.append('pos{}_act{}_{}'.format(pos, act, self.id))

                for vel in range(self.vel.shape[0] - 1):
                    for act in range(self.dm_act):
                        param_names.append('vel{}_act{}_{}'.format(vel, act, self.id))

                for t in range(1, self.times.shape[0]):
                    param_names.append('t{}_{}'.format(t, self.id))

        else:
            if self.id is None:
                for pos in range(self.pos.shape[0]):
                    for act in range(self.dm_act):
                        param_names.append('pos{}_act{}'.format(pos, act))

                for vel in range(self.vel.shape[0]):
                    for act in range(self.dm_act):
                        param_names.append('vel{}_act{}'.format(vel, act))

                for t in range(1, self.times.shape[0]):
                    param_names.append('t{}'.format(t))
            else:
                for pos in range(self.pos.shape[0]):
                    for act in range(self.dm_act):
                        param_names.append('pos{}_act{}_{}'.format(pos, act, self.id))

                for vel in range(self.vel.shape[0]):
                    for act in range(self.dm_act):
                        param_names.append('vel{}_act{}_{}'.format(vel, act, self.id))

                for t in range(1, self.times.shape[0]):
                    param_names.append('t{}_{}'.format(t, self.id))


        return param_names

    def get_params(self):
        if self.cyclic:
            params = np.concatenate((self.pos[:-1, :].flat, self.vel[:-1, :].flat, self.times[1:].flat))
            return params[self.param_mask]
        else:
            params = np.concatenate((self.pos.flat, self.vel.flat, self.times[1:].flat))
            return params[self.param_mask]

    def set_params(self, params):
        if self.cyclic:
            new_params = np.concatenate((self.pos[:-1, :].flat, self.vel[:-1, :].flat, self.times[1:].flat))
            new_params[self.param_mask] = params
            n_pos = self.pos[:-1, :].size
            n_vel = self.vel[:-1, :].size
            self.pos[:-1, :] = new_params[:n_pos].reshape(np.shape(self.pos[:-1, :]))
            self.vel[:-1, :] = new_params[n_pos:n_pos + n_vel].reshape(np.shape(self.vel[:-1, :]))
            self.times[1:] = new_params[n_pos + n_vel:]

            self.pos[-1] = self.pos[0]
            self.vel[-1] = self.vel[0]

        else:
            new_params = np.concatenate((self.pos.flat, self.vel.flat, self.times[1:].flat))
            new_params[self.param_mask] = params
            n_pos = self.pos.size
            n_vel = self.vel.size
            self.pos = new_params[:n_pos].reshape(np.shape(self.pos))
            self.vel = new_params[n_pos:n_pos + n_vel].reshape(np.shape(self.vel))
            self.times[1:] = new_params[n_pos + n_vel:]

        self.fit_splines()

    def set_params_by_name(self, param_vals, param_names):
        all_params = self.get_params()
        all_param_names = self.list_all_param_names()
        for i in range(len(param_names)):
            idx = None
            for j in range(len(all_param_names)):
                if all_param_names[j] == param_names[i]:
                    idx = j
            # if idx is None:
            #     raise(ValueError("unkonwn parameter name"))
            if idx is not None:
                all_params[idx] = param_vals[i]
        self.set_params(all_params)


    def set_constraints(self, constraints):
        self.constraints = constraints

    def get_mask(self, param_names):
        all_params = self.list_all_param_names()

        mask = [False for _ in range(len(all_params))]
        for param in param_names:
            for i in range(len(all_params)):
                if all_params[i] == param:
                    mask[i] = True
        return mask

    def fit_splines(self):
        for i in range(self.n_splines):
            T = self.times[i + 1] - self.times[i]
            self.spline_parameters[i, 0, :] = self.pos[i]
            self.spline_parameters[i, 1, :] = self.vel[i]
            self.spline_parameters[i, 2, :] = 3 * (self.pos[i + 1] - self.pos[i]) / T ** 2 \
                                              - (self.vel[i + 1] + 2 * self.vel[i]) / T
            self.spline_parameters[i, 3, :] = 2 * (self.pos[i] - self.pos[i + 1]) / T ** 3 \
                                              + (self.vel[i + 1] + self.vel[i]) / T ** 2

    def set_active_params(self, param_names):
        self.active_params = param_names
        self.param_mask = self.get_mask(param_names)
        self.n_params = len(param_names)


class ContextualCubicMP(CubicMP):

    def __init__(self, pos, vel, times, robot, context=None, cont_weights=None, cyclic=False, visible_params=None,
                 constraints=None, id=None, context_activation=None):

        CubicMP.__init__(self, pos, vel, times, cyclic=cyclic, visible_params=visible_params, id=id)

        if cont_weights is None:
            self.cont_weights = np.zeros(3)
        else:
            self.cont_weights = cont_weights

        self.context_activation = context_activation
        self.robot = robot
        self.context = context
        self.constraints = constraints



        self.fit_splines()

    def reset(self):
        pass

    def get_action(self, time):
        """dimensions where cont_pos is NaN will be ignored"""
        time = time % self.cycle_duration
        pos, vel, tau = super().get_action(time)
        context = self.context()

        if self.context_activation is None:
            activation = 1
        else:
            activation = np.interp(time, self.context_activation[:, 0], self.context_activation[:, 1])

        if not context is None and not self.cont_weights is None:
            jacp = self.robot.get_endeff_jacobian()
            endeff_pos = self.robot.get_endeff_position()
            vel += activation * np.linalg.pinv(jacp).dot(self.cont_weights * np.nan_to_num(context - endeff_pos))  # TODO: clip vel for real robot
            # TODO: nullspace
            # vel += jacp.T.dot(w * np.nan_to_num(cont_pos - endeff_pos))  # TODO: clip vel for real robot

        return pos, vel, tau

    def list_all_param_names(self):
        param_names = super().list_all_param_names()

        if not self.id is None:
            for dim in ['x', 'y', 'z']:
                param_names.append('cont_weight_{}_{}'.format(dim, self.id))
        else:
            for dim in ['x', 'y', 'z']:
                param_names.append('cont_weight_{}'.format(dim))
        return param_names

    def get_params(self):
        if self.cyclic:
            params = np.concatenate((self.pos[:-1, :].flat,
                                     self.vel[:-1, :].flat,
                                     self.times[1:].flat,
                                     self.cont_weights.flat))
        else:
            params = np.concatenate((self.pos.flat,
                                     self.vel.flat,
                                     self.times[1:].flat,
                                     self.cont_weights.flat))
        return params[self.param_mask]

    def set_params(self, params):
        if self.cyclic:
            new_params = np.concatenate((self.pos[:-1, :].flat,
                                         self.vel[:-1, :].flat,
                                         self.times[1:].flat,
                                         self.cont_weights.flat))
            new_params[self.param_mask] = params

            if not self.constraints is None:
                all_param_names = self.list_all_param_names()
                for i in range(len(self.constraints['equal'])):
                    idx0 = all_param_names.index(self.constraints['equal'][i][0])
                    idx1 = all_param_names.index(self.constraints['equal'][i][1])
                    new_params[idx0] = new_params[idx1]

            n_pos = self.pos[:-1, :].size
            n_vel = self.vel[:-1, :].size
            n_times = self.times[1:].size

            self.pos[:-1, :] = new_params[:n_pos].reshape(np.shape(self.pos[:-1, :]))
            self.vel[:-1, :] = new_params[n_pos:n_pos + n_vel].reshape(np.shape(self.vel[:-1, :]))
            self.times[1:] = new_params[n_pos + n_vel:n_pos + n_vel + n_times]
            self.cont_weights = new_params[n_pos + n_vel + n_times:]

            self.pos[-1] = self.pos[0]
            self.vel[-1] = self.vel[0]

        else:
            new_params = np.concatenate((self.pos.flat,
                                         self.vel.flat,
                                         self.times[1:].flat,
                                         self.cont_weights.flat))
            new_params[self.param_mask] = params

            if not self.constraints is None:
                all_param_names = self.list_all_param_names()
                for i in range(len(self.constraints['equal'])):
                    idx0 = all_param_names.index(self.constraints['equal'][i][0])
                    idx1 = all_param_names.index(self.constraints['equal'][i][1])
                    new_params[idx0] = new_params[idx1]

            n_pos = self.pos.size
            n_vel = self.vel.size
            n_times = self.times[1:].size

            self.pos = new_params[:n_pos].reshape(np.shape(self.pos))
            self.vel = new_params[n_pos:n_pos + n_vel].reshape(np.shape(self.vel))
            self.times[1:] = new_params[n_pos + n_vel:n_pos + n_vel + n_times]
            self.cont_weights = new_params[n_pos + n_vel + n_times:]

        self.fit_splines()



        # self.plot()


class ContextualCubicMP2(CubicMP):  # specialised for juggling
    t_TD_des = np.array([0.950, 0.475])
    t_TO_des = np.array([0.180, 0.655])
    b_TD_des = np.array([[ 0.075, 0.95, 0.25],
                         [-0.075, 0.95, 0.25]])

    jac = np.array([[[-8.86873923e-01, -7.76989044e-03, -4.28346513e-01, -2.10543971e-02],
                     [ 7.03560521e-02, -9.78949071e-02,  3.39844542e-02, -2.65269924e-01],
                     [ 0.00000000e+00, -8.89660232e-01,  3.21440978e-05, -3.63985308e-01]],

                    [[-8.86755134e-01,  7.92579320e-03, -4.28289707e-01,  2.14768533e-02],
                     [-7.18377488e-02, -9.78824082e-02, -3.46930179e-02, -2.65236055e-01],
                     [ 0.00000000e+00, -8.89660232e-01,  3.21440978e-05, -3.63985308e-01]]])

    def __init__(self, pos, vel, times, robot, context=None, cont_weights=None, cyclic=False, visible_params=None,
                 constraints=None, id=None, context_activation=None):

        CubicMP.__init__(self, pos, vel, times, cyclic=cyclic, visible_params=visible_params, id=id)

        if cont_weights is None:
            self.cont_weights = np.zeros(3)
        else:
            self.cont_weights = cont_weights

        self.context_activation = context_activation
        self.robot = robot
        self.context = context
        self.constraints = constraints
        self.n_balls = 2
        self.reset()

    def reset(self):
        self.att_main_gain = 1.0
        self.att_fade_in = 0.05
        self.att_fade_out = 0.05
        self.att_redirect_off = 0.0

        self.stalled_TD_predictions = self.b_TD_des.copy()
        self.fit_splines()

    def predict_touch_down_pos(self, id):
        """
        predicts time until touch down assuming parabula curve for the ball
        if the ball is below touch down height, the current position is returned
        """

        g, g2 = -9.81, 96.24
        g_vec = np.array([0., 0., g])

        pos, vel = self.robot.get_ball_states(ball_id=id)
        z, dz = pos[2], vel[2]
        if z <= self.b_TD_des[id][2]:
            t_TD = 0
        else:

            t_TD = - dz/g + np.sqrt( np.abs(dz**2/g2 - 2*(z-self.b_TD_des[id][2])/g) )

        # print('t_TD', t_TD)
        b_TD = 0.5*g_vec*t_TD**2 + vel*t_TD + pos

        return b_TD

    def get_stalled_touch_down_predictions(self, t):
        for ball_id in range(self.n_balls):
            if self.t_TO_des[ball_id-1] < t and t < self.t_TD_des[ball_id] - 0.1:
                # print('update prediction {}'.format(ball_id))
                self.stalled_TD_predictions[ball_id] = self.predict_touch_down_pos(ball_id)
        return self.stalled_TD_predictions

    def get_attention_gains(self, t):  # TODO: sigmoid for attention
        off = np.array([self.t_TD_des[0]-self.t_TO_des[1], -self.t_TO_des[0]+self.t_TD_des[0]])

        constr_times = np.array([0.0,
                                 self.att_fade_in,
                                 self.t_TD_des[1]-self.t_TO_des[0],
                                 self.t_TO_des[1]-self.t_TO_des[0],
                                 self.t_TO_des[1]-self.t_TO_des[0]+self.att_fade_out])

        constr_values = np.array([0.0,
                                  1.0,
                                  1.0,
                                  1.0 + self.att_redirect_off,
                                  0.0])

        times = (off + t) % self.cycle_duration

        attention = self.att_main_gain * np.interp(times, constr_times, constr_values)
        return attention

    def get_action(self, time):
        """dimensions where cont_pos is NaN will be ignored"""
        time = time % self.cycle_duration
        # print('\n')
        # print('time', time)
        #
        # if abs(time - self.t_TD_des[0]) < 0.002:
        #     print('TD 0')
        #
        # if abs(time - self.t_TD_des[1]) < 0.002:
        #     print('TD 1')
        #
        # if abs(time - self.t_TO_des[0]) < 0.002:
        #     print('TO 0')
        #
        # if abs(time - self.t_TO_des[1]) < 0.002:
        #     print('TO 1')

        pos, vel, tau = super().get_action(time)

        b_TD_pred = self.get_stalled_touch_down_predictions(time)
        # print('pred', b_TD_pred)
        att_gains = self.get_attention_gains(time)


        P = np.array([[1.0, 0.0, 0.0],  # projection onto xy plane
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0]])

        task_off = 0
        for i in range(self.n_balls):
            # print('task off {}'.format(i), att_gains[i] * P @ (b_TD_pred[i] - self.b_TD_des[i]))
            task_off += att_gains[i] * P @ (b_TD_pred[i] - self.b_TD_des[i])

        max_off = np.array([0.05, 0.05, 0.01])
        task_off = np.clip(task_off, -max_off, max_off)

        # print('task offs', task_off)

        jacp = (self.jac[0] + self.jac[1]) / 2
        jacp_inv = np.linalg.pinv(jacp)
        q_off = jacp_inv @ task_off

        # print('jac', jacp)
        # print('jac_inv', jacp_inv)

        # print('q_off', q_off)


        # print('taks off\n', task_off)
        # print('jac\n', jacp)
        # print('jinv\n', jacp_inv)
        # print('q_off\n', q_off)

        # TODO: clip / cale joint offset

        pos += q_off

        return pos, vel, tau

    def list_all_param_names(self):
        param_names = super().list_all_param_names()

        if not self.id is None:
            for dim in ['x', 'y', 'z']:
                param_names.append('cont_weight_{}_{}'.format(dim, self.id))
            param_names.append('att_main_{}'.format(self.id))
            param_names.append('att_redirect_off_{}'.format(self.id))
            param_names.append('att_fade_in_{}'.format(self.id))
            param_names.append('att_fade_out_{}'.format(self.id))

        else:
            for dim in ['x', 'y', 'z']:
                param_names.append('cont_weight_{}'.format(dim))
            param_names.append('att_main')
            param_names.append('att_redirect_off')
            param_names.append('att_fade_in')
            param_names.append('att_fade_out')


        return param_names

    def get_params(self):
        if self.cyclic:
            params = np.concatenate((self.pos[:-1, :].flat,
                                     self.vel[:-1, :].flat,
                                     self.times[1:].flat,
                                     self.cont_weights.flat,
                                     [self.att_main_gain, self.att_fade_in, self.att_fade_out, self.att_redirect_off]))
        else:
            params = np.concatenate((self.pos.flat,
                                     self.vel.flat,
                                     self.times[1:].flat,
                                     self.cont_weights.flat,
                                     [self.att_main_gain, self.att_fade_in, self.att_fade_out, self.att_redirect_off]))
        return params[self.param_mask]

    def set_params(self, params):
        if self.cyclic:
            new_params = np.concatenate((self.pos[:-1, :].flat,
                                         self.vel[:-1, :].flat,
                                         self.times[1:].flat,
                                         self.cont_weights.flat,
                                         [self.att_main_gain, self.att_fade_in, self.att_fade_out, self.att_redirect_off]))
            new_params[self.param_mask] = params

            if not self.constraints is None:
                all_param_names = self.list_all_param_names()
                for i in range(len(self.constraints['equal'])):
                    idx0 = all_param_names.index(self.constraints['equal'][i][0])
                    idx1 = all_param_names.index(self.constraints['equal'][i][1])
                    new_params[idx0] = new_params[idx1]

            n_pos = self.pos[:-1, :].size
            n_vel = self.vel[:-1, :].size
            n_times = self.times[1:].size
            n_cont_weight = self.cont_weights.size

            self.pos[:-1, :] = new_params[:n_pos].reshape(np.shape(self.pos[:-1, :]))
            self.vel[:-1, :] = new_params[n_pos:n_pos + n_vel].reshape(np.shape(self.vel[:-1, :]))
            self.times[1:] = new_params[n_pos + n_vel:n_pos + n_vel + n_times]
            self.cont_weights = new_params[n_pos + n_vel + n_times:n_pos + n_vel + n_times+n_cont_weight]
            off = n_pos + n_vel + n_times + n_cont_weight
            self.att_main_gain = new_params[off]
            self.att_fade_in = new_params[off+1]
            self.att_fade_out = new_params[off+2]
            self.att_redirect_off = new_params[off+3]

            self.pos[-1] = self.pos[0]
            self.vel[-1] = self.vel[0]

        else:
            new_params = np.concatenate((self.pos.flat,
                                         self.vel.flat,
                                         self.times[1:].flat,
                                         self.cont_weights.flat))
            new_params[self.param_mask] = params

            if not self.constraints is None:
                all_param_names = self.list_all_param_names()
                for i in range(len(self.constraints['equal'])):
                    idx0 = all_param_names.index(self.constraints['equal'][i][0])
                    idx1 = all_param_names.index(self.constraints['equal'][i][1])
                    new_params[idx0] = new_params[idx1]

            n_pos = self.pos.size
            n_vel = self.vel.size
            n_times = self.times[1:].size

            self.pos = new_params[:n_pos].reshape(np.shape(self.pos))
            self.vel = new_params[n_pos:n_pos + n_vel].reshape(np.shape(self.vel))
            self.times[1:] = new_params[n_pos + n_vel:n_pos + n_vel + n_times]
            self.cont_weights = new_params[n_pos + n_vel + n_times:]

        self.reset()



        # self.plot()


class DelayUntilEventMP(MovementPrimitive):

    def __init__(self, pos, event, vel=None, tau=None, trigger=None, id=None, visible_params=None):
        self.id = id
        self.pos = pos
        if vel is None:
            self.vel = np.zeros_like(self.pos)
        else:
            self.vel = vel
        if tau is None:
            self.tau = np.zeros_like(self.pos)
        else:
            self.tau = tau
        self.event = event
        self.trigger = trigger
        self.duration = np.inf
        self.has_triggered = False

        if visible_params is None:
            self.active_params = self.list_all_param_names()
        else:
            self.active_params = visible_params
        self.param_mask = self.get_mask(self.active_params)
        self.n_params = len(self.active_params)

    def reset(self):
        self.duration = np.inf
        self.has_triggered = False

    def get_action(self, time):
        if self.event():
            self.duration = time
            if not self.has_triggered and self.trigger is not None:
                self.trigger()
                self.has_triggered = True
        return self.pos, self.vel, self.tau

    def list_all_param_names(self):
        param_names = []

        for type in ['pos', 'vel', 'tau']:
            for joint in range(4):
                if self.id is None:
                    param_names.append('{}_act{}'.format(type, joint))
                else:
                    param_names.append('{}_act{}_{}'.format(type, joint, self.id))
        return param_names

    def get_params(self):
        params = np.concatenate((self.pos, self.vel, self.tau))
        return params[self.param_mask]

    def set_params(self, params):
        new_params = np.concatenate((self.pos, self.vel, self.tau))
        new_params[self.param_mask] = params
        self.pos = new_params[:4]
        self.vel = new_params[4:8]
        self.tau = new_params[8:12]

    def set_params_by_name(self, param_vals, param_names):
        all_params = self.get_params()
        all_param_names = self.list_all_param_names()
        for i in range(len(param_names)):
            idx = None
            for j in range(len(all_param_names)):
                if all_param_names[j] == param_names[i]:
                    idx = j
            # if idx is None:
            #     raise(ValueError("unkonwn parameter name"))
            if idx is not None:
                all_params[idx] = param_vals[i]
        self.set_params(all_params)

    def get_mask(self, param_names):
        all_params = self.list_all_param_names()

        mask = [False for _ in range(len(all_params))]
        for param in param_names:
            for i in range(len(all_params)):
                if all_params[i] == param:
                    mask[i] = True
        return mask

    def set_active_params(self, param_names):
        self.active_params = param_names
        self.param_mask = self.get_mask(param_names)
        self.n_params = len(param_names)

    def plot(self):
        raise NotImplementedError("{}.plot() is not implemented.".format(str(type(self))[8:-2]))


class PiecewiseMP(MovementPrimitive):

    def __init__(self, pieces, intervals, visible_params=None, constraints=None):
        self.n_pieces = len(pieces)
        self.pieces = pieces
        self.current_piece = 0
        self.sum_over_last_piece = 0.0

        if visible_params is None:
            visible_params = self.list_all_param_names()
        self.set_active_params(visible_params)

        self.set_constraints(constraints)

    def reset(self):
        for piece in self.pieces:
            piece.reset()
        self.current_piece = 0
        self.sum_over_last_piece = 0.0

    def get_action(self, time):
        # print(time)
        if time > self.sum_over_last_piece + self.pieces[self.current_piece].duration \
                and self.current_piece < len(self.pieces):
            self.sum_over_last_piece += self.pieces[self.current_piece].duration
            self.current_piece += 1
        return self.pieces[self.current_piece].get_action(time-self.sum_over_last_piece)

    def list_all_param_names(self):
        param_names = []
        for i in range(self.n_pieces):
            param_names = param_names + self.pieces[i].list_all_param_names()
        return param_names

    def get_params(self):
        return self.get_all_params()[self.param_mask]

    def get_all_params(self):
        return np.concatenate(tuple(self.pieces[i].get_params() for i in range(self.n_pieces)))

    def set_params(self, params, set_all=False):
        if set_all:
            assert len(params) == len(self.list_all_param_names())
            new_params = params
        else:
            assert len(params) == len(self.active_params)
            new_params = self.get_all_params()
            new_params[self.param_mask] = params

        if self.constraints is not None:
            all_param_names = self.list_all_param_names()
            for i in range(len(self.constraints['equal'])):
                idx0 = all_param_names.index(self.constraints['equal'][i][0])
                idx1 = all_param_names.index(self.constraints['equal'][i][1])
                new_params[idx0] = new_params[idx1]
            for i in range(len(self.constraints['offset'])):
                idx0 = all_param_names.index(self.constraints['offset'][i][0])
                idx1 = all_param_names.index(self.constraints['offset'][i][1])
                offset = self.constraints['offset'][i][2]
                new_params[idx0] = new_params[idx1] + offset
            for i in range(len(self.constraints['mirror'])):
                idx0 = all_param_names.index(self.constraints['mirror'][i][0])
                idx1 = all_param_names.index(self.constraints['mirror'][i][1])
                mirror = self.constraints['mirror'][i][2]
                new_params[idx0] = mirror - (new_params[idx1] - mirror)

        idx = [0, len(self.pieces[0].get_params())]
        self.pieces[0].set_params(new_params[idx[0]:idx[1]])
        for i in range(1, self.n_pieces):
            idx[0] = idx[1]
            idx[1] = idx[0] + len(self.pieces[i].get_params())
            self.pieces[i].set_params(new_params[idx[0]:idx[1]])

    def set_params_by_name(self, param_vals, param_names):
        for i in range(self.n_pieces):
            self.pieces[i].set_params_by_name(param_vals, param_names)

    def set_active_params(self, param_names):
        self.active_params = param_names
        self.n_params = len(param_names)
        self.param_mask = self.get_mask(param_names)

    def get_mask(self, param_names):
        all_params = self.list_all_param_names()

        mask = [False for _ in range(len(all_params))]
        for param in param_names:
            for i in range(len(all_params)):
                if all_params[i] == param:
                    mask[i] = True
        return mask

    def set_constraints(self, constraints):
        self.constraints = constraints

    def get_open_param_names(self):
        return list(np.array(self.list_all_param_names())[self.param_mask])


class Policy2bColBlind(PiecewiseMP):

    def __init__(self, params):
        poss_stroke = np.zeros((5, 4))
        vels_stroke = np.zeros_like(poss_stroke)
        times_stroke = np.linspace(1, 2, 4)

        poss_cyclic = np.zeros((4, 4))
        vels_cyclic = np.zeros_like(poss_cyclic)
        times_cyclic = np.linspace(1, 2, 4)

        policy_stroke = CubicMP(poss_stroke, vels_stroke, times_stroke, cyclic=False, id='s')
        policy_cyclic = CubicMP(poss_cyclic, vels_cyclic, times_cyclic, cyclic=True, id='c')
        intervals = np.array([policy_stroke.duration, np.inf])
        super().__init__([policy_stroke, policy_cyclic],
                         intervals)  # , visible_params=visible_params, constraints=constraints)

        self.set_params(params)

        visible_params = ['pos1_act0_s', 'pos1_act1_s', 'pos1_act3_s',
                          'pos2_act0_s', 'pos2_act1_s', 'pos2_act3_s',
                          'pos3_act0_s', 'pos3_act1_s', 'pos3_act3_s',
                          't1_s', 't2_s', 't3_s',
                          'pos0_act0_c', 'pos0_act1_c', 'pos0_act3_c',
                          'pos1_act0_c', 'pos1_act1_c', 'pos1_act3_c',
                          'pos2_act0_c',
                          'pos3_act0_c',
                          't1_c']
        self.set_active_params(visible_params)

        constraints = {'equal': [['pos4_act0_s', 'pos0_act0_c'],
                                 ['pos4_act1_s', 'pos0_act1_c'],
                                 ['pos4_act3_s', 'pos0_act3_c'],
                                 ['pos2_act1_c', 'pos0_act1_c'],
                                 ['pos2_act3_c', 'pos0_act3_c'],
                                 ['pos3_act1_c', 'pos1_act1_c'],
                                 ['pos3_act3_c', 'pos1_act3_c']],
                       'offset': [['t3_c', 't1_c', 0.475]]}
        self.set_constraints(constraints)
