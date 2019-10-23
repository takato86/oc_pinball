import numpy as np
from fourier import FourierBasis
import pdb
import os
from datetime import datetime
import copy
from scipy.special import expit, logsumexp
import time

np.random.seed(seed = 32)

class OptionCriticAgent(object):
    """
    価値関数近似：Fourier basis, https://scholarworks.umass.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1100&context=cs_faculty_pubs
    """
    def __init__(self, action_space, observation_space, n_options):
        self.action_space = action_space
        self.basis_order = 3
        self.shape_state = observation_space.shape
        self.n_options = n_options
        self.fourier_basis = FourierBasis(self.shape_state[0], observation_space, order=self.basis_order)
        self.n_features = self.fourier_basis.getNumBasisFunctions()
        # Quの初期化
        self.w_q_u = np.zeros((self.n_options, action_space.n, self.n_features)) # the number of orders
        self.options = [Option(action_space.n, observation_space.shape[0], self.n_features) for i in range(self.n_options)]
        # Qomegaの初期化
        self.w_omega = np.zeros((self.n_options, self.n_features))
        # Hyper parameters
        self.epsilon = 0.01
        self.gamma = 0.99
        self.lr_critic = 0.01
        self.lr_qlearn = 0.01
        # variables for analysis
        self.td_error_list = []
        self.td_error_list_meta = []
        self.vis_action_dist = np.zeros(action_space.n)
        self.vis_option_q = np.zeros(n_options)
    
    def set_last_q_omega(self, option, obs):
        feat = self.fourier_basis(obs)
        self.last_q_omega = self._get_q_omega_list(feat)[option]

    def act(self, observation, o):
        option = self.options[o]
        feature = self.fourier_basis(observation)
        intra_option_dist = option.get_intra_option_dist(feature)
        self.vis_action_dist = intra_option_dist
        try:
            action = np.random.choice(list(range(self.action_space.n)), 1, p=intra_option_dist)
        except ValueError:
            import pdb; pdb.set_trace()
        return action[0]

    def update(self, pre_o, pre_obs, pre_a, o, obs, a, r, done):
        """
        1. Update critic(pi_Omega: Intra Q Learning, pi_u: IntraAction Q learning)
        2. Improve actors
        """
        pre_feat = self.fourier_basis(pre_obs)
        feat = self.fourier_basis(obs)
        self._update_w_q_omega(pre_feat, pre_a, feat, r, done, pre_o, o)
        self._update_w_q_u(pre_feat, pre_a, feat, r, done, pre_o)
        q_omega = self._get_q_omega_list(feat)[o]
        v_omega = self._get_v_omega(feat)
        q_u_list = self._get_q_u_list(feat, o)
        # TODO this is baseline version.
        # q_u_list -= q_omega

        option = self.options[o]
        option.update(a, pre_feat, feat, q_u_list, q_omega, v_omega) 
    
    def _update_w_q_omega(self, pre_feat, a, feat, r, done, pre_o, o):
        """
        This is for update of intra-option learning for policy over options
        """
        update_target = r
        if not done:
            term_prob = self.options[pre_o].get_terminate(feat)
            next_q_omega_list = self._get_q_omega_list(feat)
            update_target += self.gamma * ((1 - term_prob ) * next_q_omega_list[pre_o] + term_prob * np.max(next_q_omega_list))
        td_error = update_target - self.last_q_omega
        self.td_error_list_meta.append(abs(td_error))
        grad = pre_feat # check if the dimension is #basis_order
        lr_qlearn = self.lr_qlearn / np.linalg.norm(feat)
        self.w_omega[pre_o] += lr_qlearn * td_error * grad
        if not done:
            # Very important
            self.last_q_omega = next_q_omega_list[o]

    def _update_w_q_u(self, pre_feat, a, feat, r, done, o):
        """
        This is for update of critic in option-critic
        """
        # TODO check
        update_target = r
        if not done:
            term_prob = self.options[o].get_terminate(feat)
            next_q_omega_list = self._get_q_omega_list(feat)
            update_target += self.gamma * ((1 - term_prob) * next_q_omega_list[o] + term_prob * np.max(next_q_omega_list))
        td_error = update_target - self._get_q_u_list(pre_feat, o)[a]
        self.td_error_list.append(abs(td_error))
        grad = pre_feat
        lr_critic = self.lr_critic / np.linalg.norm(feat)
        self.w_q_u[o][a] += lr_critic * td_error * grad

    def _get_q_omega_list(self, feat):
        # >> (1, #options)
        return np.dot(self.w_omega, feat)

    def _get_q_u_list(self, feat, o):
        # check
        return np.dot(self.w_q_u[o], feat)

    def get_max_q_u(self, obs, o):
        feat = self.fourier_basis(obs)
        return np.max(self._get_q_u_list(feat, o))

    def _get_v_omega(self, feat):
        """
        Policy over options is decided determistically.
        """
        q_omega_list = self._get_q_omega_list(feat)
        return np.max(q_omega_list)

    def get_option(self, obs):
        rand = np.random.rand()
        feat = self.fourier_basis(obs)
        if rand > self.epsilon:
            q_omega_list = self._get_q_omega_list(feat)
            self.vis_option_q = q_omega_list
            return np.argmax(q_omega_list)
        else:
            return np.random.choice(self.n_options)

    def get_terminate(self, obs, o):
        feat = self.fourier_basis(obs)
        return self.options[o].get_terminate(feat)

    def save_model(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, 'oc_model.npz')
        np.savez(file_path, w_q=self.w_q)
        for i, option in enumerate(self.options):
            option.save_model(os.path.join(dir_path, 'option{}.npz'.format(i+1)))

    def load_model(self, dir_path):
        file_path = os.path.join(dir_path, 'oc_model.npz')
        oc_model = np.load(file_path)
        if self._check_model(oc_model):
            self.w_q = oc_model['w_q']
        else:
            raise Exception('Not suitable model data.')
        for i, option in enumerate(self.options):
            file_path = os.path.join(dir_path, 'option{}.npz'.format(i+1))
            option.load_model(file_path)

    def _check_model(self, model):
        if model['w_q_u'].shape != self.w_q_u.shape:
            return False
        if model['w_omega'].shape != self.w_omega.shape:
            return False
        return True

class Option(object):
    def __init__(self, n_actions, n_obs, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.theta = np.zeros((n_actions, self.n_features))
        self.vartheta = np.zeros((n_features))
        self.lr_theta = 0.001 #0.001
        self.lr_vartheta = 0.001 #0.001
        self.temperature = 1.0
        # variables for analysis     

    def update(self, a, pre_feat, feat, q_u_list, q_omega, v_omega):
        """
        q_omega(obs, option), v_omega(obs, option)
        q_u_list(pre_obs, option, a)
        """
        self._update_theta(a, feat, q_u_list)
        self._update_vartheta(feat, q_omega, v_omega)
    
    def _update_theta(self, a, feat, q_u_list):
        """
        intra option policy gradient theorem
        """
        pi_theta = - self.get_intra_option_dist(feat)
        # TODO check
        pi_theta = pi_theta.reshape(1,len(pi_theta))
        feat = feat.reshape(1,len(feat))
        grad = np.multiply(pi_theta.T , feat) #温度パラメータは省略
        feat = feat.reshape(feat.shape[1], )
        grad[a] += feat
        lr_theta = self.lr_theta / np.linalg.norm(feat)
        self.theta += lr_theta * q_u_list[a] * grad

    def _update_vartheta(self, feat, q_omega, v_omega):
        """
        termination function gradient theorem
        """
        advantage = q_omega - v_omega
        beta = self.get_terminate(feat)
        lr_vartheta = self.lr_vartheta / np.linalg.norm(feat)
        self.vartheta -= lr_vartheta * advantage * feat * beta * (1 - beta)

    def get_terminate(self, feat):
        """
        linear-sigmoid functions
        """
        linear_sum = np.dot(self.vartheta, feat)
        return expit(linear_sum)

    def get_intra_option_dist(self, feat):
        """
        Boltzmann policies
        """
        energy = np.dot(self.theta, feat) # / self.temperature # >> (1, #actions)
        return np.exp(energy - logsumexp(energy))
    
    def exp(self, x):
        x = np.where(x > 709, 709, x)
        return np.exp(x)
    
    def save_model(self, file_path):
        np.savez(file_path, theta = self.theta, vartheta = self.vartheta)

    def load_model(self, file_path):
        option_model = np.load(file_path)
        if self._check_model(option_model):
            self.theta = option_model['theta']
            self.vartheta = option_model['vartheta']
        else:
            raise Exception('Not suitable model data.')

    def _check_model(self, model):
        if model['theta'].shape != self.theta.shape:
            return False
        if model['vartheta'].shape != self.vartheta.shape:
            return False
        return True

