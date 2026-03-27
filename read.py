import json
import numpy as np

class DataLoad:
    def __init__(self):
        #注意更改为所在电脑中文件的绝对地址或相对地址
        with open(r"D:\AAA_mycode\NN\data\state2_pv5z.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        q1_list = []
        q2_list = []
        q3_list = []
        energy_list = []

        for item in data:
            ps = item["parameters"]
            q1_list.append(ps["q1"])
            q2_list.append(ps["q2"])
            q3_list.append(ps["q3"])
            energy_list.append(item["energy"])

        self.q1 = np.array(q1_list)
        self.q2 = np.array(q2_list)
        self.q3 = np.array(q3_list)
        self.energy = np.array(energy_list)




    def load(self):
        X = np.column_stack((self.q1,self.q2,self.q3))
        Y = self.energy.reshape(-1, 1)
        return X, Y
