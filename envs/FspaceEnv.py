import numpy as np
import pandas as pd
import gym
import torch
import json
import random
import string
import time
import re
import nltk
nltk.download('words')
from nltk.corpus import words
from gym import spaces
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(precision=2, suppress=True)
awords = [w for w in words.words() if 5 <= len(w) <= 10]
# print(len(awords))

class FspaceEnv(gym.Env):
    def __init__(
        self,
        steps: int = 20,
        seed = 2,
        xplt = 0.99,
        threshold = 0.5,
        reward = 1,
        cwd = "/home/kylesa/avast_clf/v0.2/",
        ):
        super(FspaceEnv, self).__init__()  

        self.steps = steps
        self.threshold = threshold
        self.xplt = xplt
        self.rew = reward
        self.cwd = cwd
        self.scores = []
        self.iters = []
        random.seed(seed)

        self.total_time = 0.0
        self.num_executions = 0

        # Julia inits
        jutils = 'include(\"' + cwd + 'python_utils.jl\")'
        jl.eval(jutils)

        # Action and observation spaces
        # self.action_space = spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(64,), dtype=np.float32)

        # Load report labels
        df_labels = pd.read_csv(cwd + "data_avast/labels.csv")
        df = df_labels[df_labels.family == "vobfus"]
        self.files = [cwd + "data_avast/" + v + ".json" for v in df.iloc[:,0]]
        self.adv_files = [v for v in df.iloc[:,0]]
        # print('Total instances: ', len(self.files))
        self.advhashes = []

        # Keys that should not be modified
        self.notouchy = ["started_services", "created_services", "delete_files", "delete_keys"]

        self.resets = 0
        self.done = False
    
    def reset(self):
        # Initialize new targeted attack
        self.iter, self.adds, self.mods = 0, 0, 0
        self.resets += 1
        self.iter = 0
     
        self.done = False
        self.success = False

        self.editidx = 0 # modification indexes
        self.xpls = {} # initial set of most important features

        # Fetch new report
        # self.src = json.load(open(self.cwd + '/ptb.json'))
        # self.src = json.load(open("/home/kylesa/avast_clf/v0.2/data_avast/fffae359332148170c2cad17.json"))
        self.src = json.load(open(self.files[self.resets-1]))
        self.adv = self.src
        Main.report = self.src
        Main.threshold = self.xplt

        # Fetch most frequent calls
        self.trt = json.load(open(self.cwd + '/sum.json'))

        # Get classification and embedding for report
        # start_time = time.time()
        score, emb = jl.eval("classify2(report)")
        # elapsed_time = time.time() - start_time
        # self.total_time += elapsed_time
        # self.num_executions += 1
        # avg_time = self.total_time / self.num_executions
        # print("\rElapsed time: {:.5f} s | Avg time: {:.5f} s".format(elapsed_time, avg_time), end='', flush=True)
        self.score = np.squeeze(score)
        self.emb = np.squeeze(emb)
        self.lscore = self.score
        # print(self.score, "iteration: ", self.iter+1)
        # print(emb.shape, type(emb))
        # print(emb)

        # Classify and explain
        # scores, emb, xpl = jl.eval('xclass(report,threshold)')
        # print(scores, xpl)

        # Get explanation
        # xpl = jl.eval("explanation(report,threshold)")
        # self.xpls = xpl[0]["summary"] # initial set of most important features
        self.xpls = {}

        # Get first observation    
        # obs = [np.float32(i) for i in [0.0,0.0,1.0]]
        obs = self.emb
        # observation.append(np.float32(0.0))

        return obs

    def step(self, action):
        self.iter += 1
        # start_time = time.time()
        # Modify report
        if action == 0:
            # Add
            for key, value in self.adv["summary"].items():
                if key in self.notouchy:
                    continue
                else:
                    try:
                        self.adv["summary"][key].append(self.trt[key].pop(0))
                    except:
                        # print('empty target dict')
                        pass
            self.adds += 1
            # print("add:", time.time() - start_time)
        elif action == 1:
            # Edit
            self.edit(self.adv["summary"], self.editidx)
            self.editidx += 1
            self.mods += 1
            # print("edit:", time.time() - start_time)
        elif action == 2:
            # X-powered edit
            # Get explanation
            # start_time = time.time()
            # if not self.xpls:
            # start_time = time.time()
            xpl = jl.eval("explanation(report,threshold)")
            # elapsed_time = time.time() - start_time
            # self.total_time += elapsed_time
            # self.num_executions += 1
            # avg_time = self.total_time / self.num_executions
            # print("\rElapsed time: {:.5f} s | Avg time: {:.5f} s".format(elapsed_time, avg_time), end='', flush=True)

            # print(xpl)
            if isinstance(xpl, list):
                # self.xpls = self.merge_dicts(self.xpls, xpl[0]["summary"])
                self.xpls = xpl[0]["summary"]
            self.xedit(self.adv["summary"])
            # print("explain:", time.time() - start_time)
        else:
            print("Wrong action ID")

        # Classify report
        # start_time = time.time()
        with open(self.cwd + "curradv.json", "w") as outfile:
                json.dump(self.adv, outfile)
        jl.eval("loadrep()")
        # Main.report = self.adv
        # elapsed_time = time.time() - start_time
        # self.total_time += elapsed_time
        # self.num_executions += 1
        # avg_time = self.total_time / self.num_executions
        # print("\rElapsed time: {:.5f} s | Avg time: {:.5f} s".format(elapsed_time, avg_time), end='', flush=True)
        
        score, emb = jl.eval("classify2(report)")

    
        self.score = np.squeeze(score)
        self.emb = np.squeeze(emb)
        # print(self.score, "iteration: ", self.iter)
        # print(self.score, 'Elapsed: ', time.time() - start_time)

        # Check if report has evaded
        if self.iter >= self.steps or self.score[2] < self.threshold:
            self.done = True
            # print(self.info)
            self.scores.append(self.score[1])
            self.iters.append(self.iter)
            # Write adversarial report to JSON
            # with open(self.cwd + "data_adv/" + self.adv_files[self.resets-1] + "_adv.json", "w") as outfile:
            #     json.dump(self.adv, outfile)
            # self.advhashes.append([self.adv_files[self.resets-1], 'vobfus'])
            # Main.GC.gc()
    
        # Get observation
        # obs = [np.float32(i) for i in [0.0,0.0,1.0]]
        obs = self.emb

        # Get reward
        r = self.reward(self.rew)
        self.lscore = self.score

        # Get info
        info = {"resets" : self.resets,
                "iterations" : self.iter,
                "adds" : self.adds,
                "mods" : self.mods,
                "score" : self.score,
                "reward" : r}
        self.info = info
        # print(info)
    
        return obs, r, self.done, info
    
    def edit(self, raport, editidx):
        for key, value in raport.items():
            if key in ["keys", "read_keys", "write_keys", "delete_keys"]:
                if len(value) > editidx: # modify only if more entries are available
                    v = value[editidx]
                    # if len(v.split("\\")) > 1:
                    #     ln = len(v.split("\\"))
                    #     original = "\\".join(v.split("\\")[-ln:])
                    #     nv = v.replace(original, "".join(random.choices(string.ascii_letters + string.digits, k=5)) + "\\" + "".join(random.choices(string.ascii_letters + string.digits, k=5)))
                    # else:
                    #     nv = v.replace(v.split("\\")[-1], "".join(random.choices(string.ascii_letters + string.digits, k=5)) + "\\" + "".join(random.choices(string.ascii_letters + string.digits, k=5)))
                    nv = self.replace_words(v)
                    value[editidx] = nv
                    # raport[key] = value
            elif key in ["resolved_apis"]:
                # if len(value) > editidx: # modify only if more entries are available
                #     # uphold api format in modification: xxx.dll.yyy
                #     if len(value) >= editidx:
                #         nv = "".join(random.choices(string.ascii_letters + string.digits, k=5)) + ".dll." + "".join(random.choices(string.ascii_letters + string.digits, k=5))
                #     value[editidx] = nv
                #     # raport[key] = value
                pass
            else:
                pass
        return raport
    
    def xedit(self, raport):
        # Iterate through each key-value pair in explainer features
        # print(raport, '\n\n')
        for key, arr in self.xpls.items():
            if key == 'resolved_apis':
                continue 
            for entry in arr:
                # if entry in raport[key]:
                # print(raport[key],'\n')
                # print(key, arr,'\n')
                index = raport[key].index(entry)
                self.mods += 1
                # if len(entry.split("\\")) > 1:
                #     # mb instead of random string, just remove (like a scupltor does on marble)
                #     og = "\\".join(entry.split("\\")[-2:])
                #     nv = entry.replace(og, "".join(random.choices(string.ascii_letters + string.digits, k=5)) + "\\" + "".join(random.choices(string.ascii_letters + string.digits, k=5)))
                # else:
                #     nv = entry.replace(entry.split("\\")[-1], "".join(random.choices(string.ascii_letters + string.digits, k=5)) + "\\" + "".join(random.choices(string.ascii_letters + string.digits, k=5)))
                nv = self.replace_words(entry)
                raport[key][index] = nv
        # Remove all entries from xpls
        self.xpls = {}
    
    def merge_dicts(self, dict1, dict2):
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result:
                for v in value:
                    if v not in result[key]:
                        result[key].extend(value)
            else:
                result[key] = value
        return result
        
    def observation(self):
        obs = 0
        return obs

    def reward1(self):
        return self.lscore[2] - self.score[2]

    def reward2(self):
        return self.reward1 - 0.02

    def reward(self, reward_nr):
        if reward_nr == 1:
            # R1
            reward = self.reward1()
        if reward_nr == 2:
            # R2
            reward = self.reward2()
        return reward

    def printrep(self):
        print(self.adv)
    
    def replace_words(self, s):
        parts = s.split('\\')
        if len(parts) > 2:
            # Find the index of the third element
            index = 2 + len(re.findall('\\\\', s[:s.index(parts[2])]))
            # Replace words from the index
            for i in range(index, len(parts)):
                # Generate a random word from the dictionary
                parts[i] = random.choice(awords)
            return '\\'.join(parts)
        else:
             a = random.choice(awords)
             b = random.choice(awords)
             return a + '\\' + b

    def write_hashes(self):
        # df = pd.DataFrame(self.advhashes, columns = ['hash', 'family'])
        # df.to_csv(self.cwd + "data_adv/" + "labels_adv.csv", index=False)
        print("avg target score: ", np.mean(self.scores))
        print("avg length: ", np.mean(self.iters))