import numpy as np
import pandas as pd
import gym
import torch
import json
import random
import string
import nltk
nltk.download('words')
from nltk.corpus import words
from gym import spaces
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
from utils import repstats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cwd = "/home/kylesa/avast_clf/v0.2/"
settings = {
    "corpus": cwd + "/corpus.json",
    "target_vals": cwd + "/sum.json",
    "labels": cwd + "data_avast/labels.csv",
    "report_folder": cwd + "data_avast/",
    "adv_folder": cwd + "data_smp/"}

# Load corpus from JSON file
with open(settings["corpus"], 'r') as f:
    corpus = json.load(f)
# corpus = [string for string in corpus if ':' not in string and len(string)<30]
# Keep the top 5K keys
corpus = dict(list(corpus.items())[:5000])
# Load english corpus
awords = [w for w in words.words() if 4 <= len(w) <= 12]
# Separator used for file paths
sep = '//'

class PspaceEnv(gym.Env):
    def __init__(
        self,
        adv = True,
        steps: int = 100,
        train = True,
        seed = 2,
        xplt = 0.95,
        threshold = 0.5,
        reward = 3,
        ):
        super(PspaceEnv, self).__init__()  

        self.steps = steps
        self.threshold = threshold
        self.xplt = xplt
        self.train = train
        self.rew = reward
        self.cwd = cwd
        self.stats = repstats()
        self.scores = []
        self.iters = []
        self.exps = []
        self.rews = []
        Main.advflag = adv
        random.seed(seed)

        self.total_time = 0.0
        self.num_executions = 0

        # Julia inits
        jutils = 'include(\"' + cwd + 'python_utils.jl\")'
        jl.eval(jutils)

        # Action and observation spaces
        self.action_space = spaces.MultiDiscrete([3,13,4])
        self.observation_space = spaces.Box(low=0, high=1, shape=(32,), dtype=np.float32)

        # Load report labels
        df_labels = pd.read_csv(settings["labels"])
        df = df_labels[df_labels.family == "vobfus"]
        self.files = [cwd + "data_avast/" + v + ".json" for v in df.iloc[:,0]]
        self.adv_files = [v for v in df.iloc[:,0]]
        # print('Total instances: ', len(self.files))
        self.advhashes = []

        # Key dict
        self.keydict = {0: "files",
                        1: "read_files",
                        2: "write_files",
                        3: "delete_files",
                        4: "keys",
                        5: "read_keys",
                        6: "write_keys",
                        7: "delete_keys",
                        8: "executed_commands",
                        9: "resolved_apis",
                        10: "mutexes",
                        11: "created_services",
                        12: "started_services"}

        self.resets = 0
        self.done = False
    
    def reset(self):
        # Initialize new targeted attack
        self.iter, self.adds, self.mods = 0, 0, 0
        self.resets += 1
        self.accrew = 0
     
        self.done = False
        self.success = False

        # Load target class entries
        self.trt = json.load(open(settings["target_vals"]))
        # Fetch new report
        if self.train:
            self.src = json.load(open(self.files[self.resets-1]))
        else:
            self.src = json.load(open(self.files[len(self.files)-self.resets]))
        self.stats.add_size(self.src['summary'], True)
        self.adv = self.src
        # print(sum(len(lst) for lst in self.adv.values()))
        Main.report = self.src
        Main.threshold = self.xplt

        # Define max modification indexes and initial set of important feats
        self.size = [len(v) for key, v in self.adv["summary"].items()]
        self.editidx = np.zeros(len(self.keydict)).astype(int)
        self.xpls = {}

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

        # Get first observation    
        obs = self.emb

        return obs

    def step(self, action):
        self.iter += 1
        self.penalty = -0.1
        self.stats.add_actions(action.tolist())
        # Modify report
        # print(action)
        if action[0] == 0:
            # Add
            key = self.keydict[action[1]]
            # for i, j in trt.items():
            #     print(i, len(j))
            if len(self.trt[key]) > 0:
                self.adv["summary"][key].append(self.trt[key].pop(0))
                self.adds += 1
            # else:
            #     self.penalty -= 0.1
        elif action[0] == 1:
            # Edit
            self.edit(self.adv["summary"], action[1], action[2])
            # print("edit:", time.time() - start_time)
        elif action[0] == 2:
            # X-edit
            # Get explanation
            xpl = jl.eval("explanation(report,threshold)")
            if isinstance(xpl, list):
                self.xpls = xpl[0]["summary"]
            self.xedit(self.adv["summary"], action[2])
            # print("explain:", time.time() - start_time)
        else:
            print("Wrong action ID")

        # Classify report
        with open(self.cwd + "curradv.json", "w") as outfile:
                json.dump(self.adv, outfile)
        jl.eval("loadrep()")
        
        score, emb = jl.eval("classify2(report)")

    
        self.score = np.squeeze(score)
        self.emb = np.squeeze(emb)

        # Check if report has evaded
        if self.iter >= self.steps or self.score[2] < self.threshold:
            # print(self.score[2], self.iter)
            self.done = True
            self.scores.append(self.score[1])
            self.iters.append(self.mods + self.adds)
            self.rews.append(self.accrew)
            self.stats.add_size(self.adv["summary"], False)
            self.stats.add_exps(self.exps)
            # Write adversarial report to JSON
            with open(settings["adv_folder"]+ self.adv_files[self.resets-1] + "_adv.json", "w") as outfile:
                json.dump(self.adv, outfile)
            self.advhashes.append([self.adv_files[self.resets-1], 'vobfus'])
    
        # Get observation
        obs = self.emb

        # Get reward
        r = self.reward(self.rew)
        self.accrew += r
        # print(r)
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
    
    def edit(self, raport, key, gen):
        name = self.keydict[key]
        if name == "resolved_apis":
            self.penalty -= 1
        else:
            # modify only if more entries are available
            pos = self.editidx[key]
            if pos < self.size[key]:
                # print(key,pos,raport[name])
                entry = raport[name][pos]
                entry = self.replace_words(entry, gen)
                raport[name][pos] = entry

            self.editidx[key] += 1
            self.mods += 1
        return raport
    
    def xedit(self, raport, gen):
        # Iterate through each key-value pair in explainer features
        # print(len(self.xpls.items()))
        for key, arr in self.xpls.items():
            self.exps.append(key)
            # print(self.iter, key)
            if key == 'resolved_apis':
                continue 
            for entry in arr:
                index = raport[key].index(entry)
                self.mods += 1
                nv = self.replace_words(entry, gen)
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
        # add 
        obs = 0
        return obs

    def reward1(self):
        return self.lscore[2] - self.score[2]

    def reward2(self):
        return self.reward1() + self.penalty
    
    def reward3(self):
        return self.reward1() * 10 + self.penalty

    def reward(self, reward_nr):
        if reward_nr == 1:
            reward = self.reward1()
        if reward_nr == 2:
            reward = self.reward2()
        if reward_nr == 3:
            reward = self.reward3()
        return reward

    def printrep(self):
        print(self.adv)
    
    def replace_words(self, w, mode):
        # for sep in ["\\","/"]:
            # if sep in w:
        parts = w.split(sep)
        if len(parts) > 2:
            # Replace words from the index
            for i in range(2, len(parts)):
                parts[i] = self.get_word(mode)
            return sep.join(parts)
        elif len(parts) == 1:
            return self.get_word(mode)
        else:
            return self.get_word(mode) + sep + self.get_word(mode)
        
    def get_word(self, mode):
        if mode == 3:
            mode = random.choice([0,1,2])
        if mode == 0:
            # English corpus
            word = random.choice(awords)
        elif mode == 1:
            # Sample from top target keys, weighted by their frequency
            word = random.choices(list(corpus.keys()), weights=list(corpus.values()), k=1)[0]
        elif mode == 2:
            # Random string
            lgth = random.randrange(4, 10)
            word = ''.join(random.choices(string.ascii_letters + string.digits, k=lgth))
        # print(word)
        return word

    def retrain(self):
        cacc, racc = jl.eval("retrain()")
        return cacc, racc

    def write_hashes(self):
        df = pd.DataFrame(self.advhashes, columns = ['hash', 'family'])
        df.to_csv(settings["adv_folder"] + "labels_adv.csv", index=False)
        
    def summary(self):    
        asc = np.mean(self.scores)
        aln = np.mean(self.iters)
        arw = np.mean(self.rews)
        rst = self.resets
        act = self.stats.get_acts()
        xps = self.stats.get_exps()
        stat = self.stats.get_stats()
        return [asc, aln, arw, rst], [act, xps, stat]