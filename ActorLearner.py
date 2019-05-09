import tensorflow as tf
import numpy as np
from PIL import Image
import gym
from Agent import ActorCriticAgent
import threading
import time

class StateHolder():
    def __init__(self, num_states, do_preprocess):
        self.num_states = num_states
        self.do_preprocess = do_preprocess
        self.reset()
        
    def reset(self):
        self.states = []
        self.states_raw = []

    def preprocess(self, state):
        # Resize and convert to gray scale
        new_state = np.array(Image.fromarray(state).resize((94, 114), Image.ANTIALIAS).convert("L"))
        # Crop
        new_state = new_state[15:-5,3:-3]

        new_state = (new_state / 255).astype(np.float32)
        return new_state
    
    def add_state(self, state):
        self.states_raw.append(state)
        if self.do_preprocess:
            state = self.preprocess(state)
        
        self.states.append(state)
        
        if len(self.states) > self.num_states * 5:
            self.states = self.states[-self.num_states - 1:]
            self.states_raw = self.states_raw[-self.num_states - 1:]
    
    def get_state(self):
        return_states = []

        for i in range(self.num_states):
            source_index = len(self.states) - self.num_states + i
            
            if source_index < 0:
                return_states.append((np.zeros_like(self.states[0])).astype(np.float32))
            else:
                return_states.append(self.states[source_index])
        
        return np.array(return_states).transpose([1, 2, 0])


class ActorLearner():
    def __init__(self, sess, get_global_param_func, num_states, num_steps_per_train, discount_rate, scope_name, thread_id, report_callback, lock, optimizer, lr_placeholder, skip_frame, device="/CPU:0"):
        self.get_global_param_func = get_global_param_func
        self.num_steps_per_train = num_steps_per_train
        self.scope_name = scope_name
        self.sess = sess
        self.skip_frame = skip_frame

        self.thread_id = thread_id
        self.lock = lock
        self.report_callback = report_callback

        self.report_freq = 2000

        self.state_holder = StateHolder(num_states, True)
        
        self.env = gym.make("Breakout-v0")
        self.env.env.frameskip = 4
        num_action = self.env.action_space.n

        # Check preprocessed state shape
        state = self.env.reset()
        self.state_holder.add_state(state)
        input_shape = list(self.state_holder.get_state().shape)

        with tf.device(device):
            with tf.variable_scope(self.scope_name):
                self.agent = ActorCriticAgent(self.sess, input_shape, num_action, discount_rate, self.scope_name, optimizer, lr_placeholder)
        
    def prepare_for_run(self, local_step):
        self.local_step = local_step
        self.local_step_prev = self.local_step
        self.ready = True
    
    def run(self):
        if not self.ready: assert "Not ready"
        
        global_step = 0

        while True:
            
            self.state_holder.reset()
            state = self.env.reset()
            self.state_holder.add_state(state)

            episode_reward = 0
            loss_sum = 0
            policy_loss_sum = 0
            value_loss_sum = 0
            entropy_sum = 0
            num_loss_sum = 0

            sync_time = 0
            exp_accum_time = 0
            train_time = 0
            start_time = time.time()

            step = 0
            last_action = 0

            self.agent.sync_parameter()

            while True:
                sync_start_time = time.time()
                global_step_new = self.get_global_param_func()
                if global_step_new > global_step: global_step = global_step_new
                
                if global_step % 1 == 0:
                    self.agent.sync_parameter()
                sync_time += time.time() - sync_start_time
                
                exp_accum_start_time = time.time()
                states = []
                actions = []
                rewards = []
                terminals = []
                
                for _ in range(self.num_steps_per_train * self.skip_frame):
                    state = self.state_holder.get_state()

                    if step % self.skip_frame == 0:
                        action = self.agent.predict_action_with_epsilon_greedy(state, self.calc_epsilon(global_step))
                        last_action = action
                    else:
                        action = last_action
                    
                    state_next, reward, terminal, _ = self.env.step(action)
                    if reward > 1.0: reward = 1.0
                    episode_reward += reward
                    
                    if step % self.skip_frame == 0:
                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        terminals.append(terminal)

                        self.state_holder.add_state(state_next)
                    
                        self.local_step += 1
                        global_step += 1
                    
                    step += 1

                    if terminal: break
                exp_accum_time += time.time() - exp_accum_start_time
                
                train_start_time = time.time()
                #self.lock.acquire()
                loss, pl, vl, en = self.agent.train(states, actions, rewards, terminals, 0.0001)
                #self.lock.release()

                loss_sum += loss
                policy_loss_sum += pl
                value_loss_sum += vl
                entropy_sum += en

                num_loss_sum += 1
                
                train_time += time.time() - train_start_time

                if terminal: break

            print("EpReward " + str(episode_reward) + ", Loss " + format(loss_sum / num_loss_sum, ".6f") + "(" + format(policy_loss_sum / num_loss_sum, ".3f") + " " + format(value_loss_sum / num_loss_sum, ".3f") + " " + format(entropy_sum / num_loss_sum, ".3f") + "), " + self.scope_name + ", " + str(self.local_step) + ", " + str(global_step) + ", " + format(sync_time, ".2f") + " " + format(exp_accum_time, ".2f") + " " + format(train_time, ".2f") + " " + format(time.time() - start_time, ".2f"))

            if self.local_step - self.local_step_prev > self.report_freq:
                self.report_callback(self.thread_id, self.local_step - self.local_step_prev, None)
                self.local_step_prev = self.local_step

            
    def calc_epsilon(self, step):
        epsilon_end_step = 4000000
        initial_epsilon = 1.0

        rand = np.random.random()

        if rand <= 0.4:
            end_epsilon = 0.1
        elif 0.4 < rand and rand <= 0.7:
            end_epsilon = 0.01
        else:
            end_epsilon = 0.5
        
        epsilon = initial_epsilon - (initial_epsilon - end_epsilon) * (step / epsilon_end_step)

        return max(epsilon, end_epsilon)





        
        
        

        

        
    
