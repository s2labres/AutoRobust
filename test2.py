import os
import numpy as np

# cwd = '/home/kylesa/avast_clf/v0.2/data_smp/'

# pta = os.listdir(cwd)
# for item in pta:
#     if item.endswith(".json"):
#         os.remove(os.path.join(cwd, item))

# a = np.array([[1,1,0],[0,0,2],[3,3,1]])
# unique, counts = np.unique(a[:,0], return_counts=True)
# x = np.asarray((unique, counts)).T
# print(x)

sz = {"files":["C:\\Windows\\Globalization\\Sorting\\sortdefault.nls","\\Device\\KsecDD","C:\\Users\\John\\AppData\\Local\\Temp\\ffe27774e7140d05acd0681e.exe.cfg","C:\\Windows\\sysnative\\C_932.NLS","C:\\Windows\\sysnative\\C_949.NLS","C:\\Windows\\sysnative\\C_950.NLS","C:\\Windows\\sysnative\\C_936.NLS","C:\\Users\\John\\AppData\\Local\\Temp\\ffe27774e7140d05acd0681e.exe","C:\\Users\\John\\joqig.exe","\\??\\MountPointManager","C:\\Users\\John\\AppData\\Local\\Temp\\wsock32.DLL","C:\\Windows\\System32\\wsock32.dll","\\Device\\Afd\\AsyncSelectHlp","C:\\Users\\John\\joqig.exe.cfg"],"read_files":["C:\\Windows\\Globalization\\Sorting\\sortdefault.nls","\\Device\\KsecDD","C:\\Users\\John\\AppData\\Local\\Temp\\ffe27774e7140d05acd0681e.exe","C:\\Windows\\System32\\wsock32.dll","\\Device\\Afd\\AsyncSelectHlp","C:\\Users\\John\\joqig.exe"],"write_files":["C:\\Users\\John\\joqig.exe","\\Device\\Afd\\AsyncSelectHlp"],"delete_files":[]}
size = sum(len(lst) for lst in sz.values())
print(size)