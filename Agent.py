import tensorflow as tf
import numpy as np
import time
from PIL import Image

slim = tf.contrib.slim

class ActorCriticAgent():
    def __init__(self, sess, input_shape, num_actions, discount_rate, scope_name, optimizer=None, lr_placeholder=None):
        # input_shape should shape [height, width, num state history]
        self.sess = sess
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.discount_rate = discount_rate
        self.scope_name = scope_name

        self.build_prediction_network()
        self.build_training_operator(optimizer, lr_placeholder)
        self.build_sync_parameter_operator()
        self.build_apply_gradient_operator()
        self.summary = tf.summary.merge_all()
    
    def build_prediction_network(self):
        with tf.variable_scope("prediction_network"):
            self.state_placeholder = tf.placeholder(tf.float32, [None] + self.input_shape, name="state")
            self.is_train_placeholder = tf.placeholder(tf.bool, name="is_train")

            self.conv_0 = slim.conv2d(self.state_placeholder, 16, [8, 8], 4, scope="conv_0")
            self.conv_1 = slim.conv2d(self.conv_0, 32, [4, 4], 2, scope="conv_1")
            
            flatten = slim.flatten(self.conv_1)
            
            self.fc_0 = slim.fully_connected(flatten, 256, scope="fc_0")
            
            self.fc_1 = slim.fully_connected(self.fc_0, self.num_actions + 1, activation_fn=None, scope="fc_1")

            self.policy = slim.softmax(self.fc_1[:,:self.num_actions])
            self.value = tf.reshape(self.fc_1[:,-1], [-1])

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name + "/prediction_network")
        self.var_list_global = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global_agent/prediction_network")

        
    def build_training_operator(self, optimizer, lr_placeholder):
        with tf.variable_scope("training_operator"):
            self.reward_placeholder = tf.placeholder(tf.float32, [None], name="reward")
            self.action_placeholder = tf.placeholder(tf.int32, [None], name="action")
            self.value_placeholder = tf.placeholder(tf.float32, [None], name="value")

            if lr_placeholder == None:
                self.learning_rate_placeholder = tf.placeholder(tf.float32, name="learning_rate")
            else:
                self.learning_rate_placeholder = lr_placeholder
            
            policy_clipped = tf.clip_by_value(self.policy, 1e-20, 1.0)

            # self.policy shape is [batch, num_actions]
            # selected_policy and self.value shape is [batch]
            policy_flat = tf.reshape(policy_clipped, [-1])
            action_index = tf.range(tf.shape(self.policy)[0]) * tf.shape(self.policy)[1]
            action_index = action_index + self.action_placeholder
            self.selected_policy = tf.gather(policy_flat, action_index)

            self.entropy = tf.reduce_mean(policy_clipped * tf.log(policy_clipped))
            
            self.advantage = self.reward_placeholder - self.value_placeholder
            #self.advantage = tf.maximum(self.advantage, 0)

            self.policy_loss = -tf.reduce_mean(tf.log(self.selected_policy) * self.advantage)
            self.value_loss = tf.reduce_mean((self.reward_placeholder - self.value)**2)
            self.loss = self.policy_loss + self.value_loss + self.entropy * 0.1

            self.entropy_2 = tf.reduce_mean(policy_clipped * tf.log(policy_clipped), axis=1)
            self.policy_loss_2 = -tf.log(self.selected_policy) * self.advantage
            self.value_loss_2 = (self.reward_placeholder - self.value)**2
            self.loss_2 = self.policy_loss_2 + self.value_loss_2 + self.entropy_2 * 0.1
            
            tf.summary.scalar("policy_loss", self.policy_loss)
            tf.summary.scalar("value_loss", self.value_loss)

            #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_placeholder)
            if optimizer == None:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
            else:
                self.optimizer = optimizer

            self.gradient = tf.gradients(self.loss, self.var_list)
            self.gradient_2 = tf.gradients(self.loss_2, self.var_list)

    def build_training_operator2(self, optimizer, lr_placeholder):
        with tf.variable_scope("training_operator"):
            self.reward_placeholder = tf.placeholder(tf.float32, name="reward")
            self.action_placeholder = tf.placeholder(tf.int32, name="action")
            self.terminal_placeholder = tf.placeholder(tf.bool, name="terminal")

            values = tf.Variable(self.value, trainable=False)


    def build_sync_parameter_operator(self):
        if len(self.var_list_global) != len(self.var_list):
            assert "Variable length unmatch"

        assign_op = []
        for g, l in zip(self.var_list_global, self.var_list):
            if g.shape != l.shape:
                assert "Shape unmatch"
            assign_op.append(l.assign(g))

        self.sync_global_params_op = assign_op

    def build_apply_gradient_operator(self):
        grads_and_vars = []
        for i, param in enumerate(self.var_list_global):
            # gradient is list of tuple of (gradient, variable)
            #grads_and_vars.append((self.gradient[i][0], param))
            grads_and_vars.append((self.gradient[i], param))

        self.apply_gradient_op = self.optimizer.apply_gradients(grads_and_vars)

    def predict_action(self, state):
        # Predict logit for single state
        logit = self.sess.run(self.policy, feed_dict={self.state_placeholder:[state], self.is_train_placeholder:False})
        # Return index of max logit
        return np.argmax(logit[0])
    
    def predict_action_with_epsilon_greedy(self, state, epsilon):
        if np.random.random() > epsilon:
            action = self.predict_action(state)
            return action
        else:
            return np.random.randint(self.num_actions)

    def sync_parameter(self):
        self.sess.run(self.sync_global_params_op)

    def train(self, states, actions, rewards, terminals, learning_rate):
        if len(states) <= 1:
            return 0, 0, 0, 0
        
        if np.random.random() < 0.2:
            if sum(rewards) == 0 and sum(terminals) == 0:
                return 0, 0, 0, 0

        values = self.sess.run(self.value, feed_dict={self.state_placeholder:states, self.is_train_placeholder:True})
        
        R = [0 for _ in range(len(states))]
        if not terminals[-1]:
            R[-1] = values[-1]
        
        for i in range(len(states) - 2, -1, -1):# loop from t-1 to 0
            R[i] = rewards[i] + self.discount_rate * R[i + 1]
            
        length = len(states) - 1
        feed_dict = {
            self.state_placeholder: states[:length],
            self.reward_placeholder: R[:length],
            self.learning_rate_placeholder: learning_rate,
            self.action_placeholder: actions[:length],
            self.value_placeholder: values[:length],
            self.is_train_placeholder: True
        }

        _, loss, pl, vl, en, grad, p = self.sess.run(
            [self.apply_gradient_op, self.loss, self.policy_loss, self.value_loss, self.entropy, self.gradient_2, self.policy], 
            feed_dict=feed_dict)

        if np.isnan(loss):
            assert "NaN!"

        return loss, pl, vl, en
