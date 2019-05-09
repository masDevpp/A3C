import os
import tensorflow as tf
import numpy as np
from Agent import ActorCriticAgent
from ActorLearner import StateHolder, ActorLearner
import gym
import time
import threading
from PIL import Image

class Host():
    def __init__(self, num_states, discount_rate, num_steps_per_train, num_threads, log_dir):
        self.env = gym.make("Breakout-v0")
        self.env.env.frameskip = 4
        self.evaluating = False
        self.num_states = num_states
        num_actions = self.env.action_space.n
        state_holder = StateHolder(num_states, True)
        state_holder.add_state(self.env.reset())
        input_shape = list(state_holder.get_state().shape)
        self.skip_frame = 1

        self.num_threads = num_threads
        self.log_dir = log_dir
        
        #self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        with tf.variable_scope("global_agent"):
            self.global_agent = ActorCriticAgent(self.sess, input_shape, num_actions, discount_rate, "global_agent")
            optimizer = self.global_agent.optimizer
            lr_placeholder = self.global_agent.learning_rate_placeholder
        
        self.learners = []
        self.threads = []
        self.local_steps = []
        self.lock = threading.Lock()
        print("Build learners")
        for i in range(self.num_threads):
            device = "/CPU:0"
            if i == 0: device = "" # GPU will automatically assigned for some learner
            self.learners.append(ActorLearner(self.sess, self.get_global_param, num_states, num_steps_per_train, discount_rate, "learner" + str(i), i, self.status_report_callback, self.lock, optimizer, lr_placeholder, self.skip_frame, device=device))
            self.local_steps.append(0)
        
        self.global_step_variable = tf.Variable(0, trainable=False, name="global_step")
        self.local_steps_variable = tf.Variable(self.local_steps, trainable=False, name="local_steps")
        self.best_eval_score_variable = tf.Variable(150.0, trainable=False)

        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.saver = tf.train.Saver()
        self.saver2 = tf.train.Saver(max_to_keep=1000)

        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Load checkpoint " + ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Initialize variables")
            self.sess.run(tf.global_variables_initializer())

        self.global_step = self.sess.run(self.global_step_variable)
        self.local_steps = self.sess.run(self.local_steps_variable)
        self.best_eval_score = self.sess.run(self.best_eval_score_variable)
        self.global_step_prev = self.global_step
        self.save_frequency = 3000 * len(self.learners)

        print("Invoke threads")
        for i in range(len(self.learners)):
            self.learners[i].prepare_for_run(self.local_steps[i])
            self.threads.append(threading.Thread(target=self.learners[i].run))
            self.threads[i].start()

        print("Threads invoked")

    def get_global_param(self):
        return self.global_step
    
    def status_report_callback(self, thread_id, elapse_step, summary):
        self.global_step += elapse_step
        self.local_steps[thread_id] += elapse_step
        
        if self.global_step - self.global_step_prev > self.save_frequency:
            self.global_step_prev = self.global_step
            print("GlobalStep " + str(self.global_step))

            self.sess.run(self.global_step_variable.assign(self.global_step))
            self.sess.run(self.local_steps_variable.assign(self.local_steps))
            
            try:
                self.saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"), global_step = self.global_step)
                if summary != None:
                    self.summary_writer.add_summary(summary, self.global_step)
            except:
                print("Progress save fail " + str(self.global_step))
            
            try:
                if not self.evaluating:
                    self.evaluating = True
                    eval_reward, images = self.evaluation(self.env, self.num_states, self.global_agent, self.skip_frame)
                    print("EvalReward " + str(eval_reward))
                    
                    with open(os.path.join(self.log_dir, "log.txt"), "a") as f:
                        f.write("GlobalStep " + str(self.global_step) + ", EvalReward " + str(eval_reward) + ", time" + format(time.time(), ".0f") + ", " + time.asctime() + "\n")

                    if eval_reward > self.best_eval_score:
                        img = self.env.render("rgb_array")
                        img = Image.fromarray(img)
                        img.save(os.path.join(self.log_dir, "save", str(eval_reward) + "_" + str(int(time.time()) % 100) + ".jpg"))
                        images[0].save(os.path.join(self.log_dir, "save", str(eval_reward) + "_" + str(int(time.time()) % 100) + ".gif"), save_all=True, append_images=images[1:])
                        
                        self.best_eval_score = (eval_reward + self.best_eval_score * 7) / 8
                        self.sess.run(self.best_eval_score_variable.assign(self.best_eval_score))
                        self.saver2.save(self.sess, os.path.join(self.log_dir, "save", "model.ckpt" + str(eval_reward)), global_step = self.global_step)
            except:
                print("Eval fail")
            finally:
                self.evaluating = False
            
    
    def evaluation(self, env, num_states, agent, skip_frame, step=3000):
        
        state_holder = StateHolder(num_states, True)
        state_holder.reset()
        state = env.reset()
        state_holder.add_state(state)
        eval_reward = 0
        last_action = 0

        images = []

        for i in range(step):
            if i % 3 == 0:
                rgb = env.render("rgb_array")
                images.append(Image.fromarray(rgb))

            if i % skip_frame == 0:
                s = state_holder.get_state()
                action = agent.predict_action(s)
                last_action = action
            else:
                action = last_action
            
            state_next, reward, terminal, _ = env.step(action)
            
            if i % skip_frame == 0:
                state_holder.add_state(state_next)
            
            eval_reward += reward

            if terminal:break
        
        rgb = env.render("rgb_array")
        images.append(Image.fromarray(rgb))
        #env.close()
        return eval_reward, images

def main():
    num_states = 4
    discount_rate = 0.99
    num_steps_per_train = 5
    num_threads = 16#32#16
    log_dir = os.path.join(os.getcwd(), "log")

    host = Host(num_states, discount_rate, num_steps_per_train, num_threads, log_dir)




if __name__ == "__main__":
    main()