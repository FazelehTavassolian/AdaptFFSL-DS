from pathlib import Path
import torch
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


root = Path('./experimental/Runresult')

rs = []
for p in root.glob("*.pt"):
    mName = p.stem[:-3]
    loads = torch.load(p)
    rs.append({'accuracy': loads['rnd_vl_acc'], 'loss': loads['rnd_vl_loss'],'name':mName,'W':loads['model']})

font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 70}

matplotlib.rc('font', **font)
#
#print('reward:\n',loads['reward'])
print('*************************************************')
#print('Q-Value:\n',loads['q_value'])
print('*************************************************')
#print('delay:\n',loads['delay'])
#print('sum delay:\n',np.sum(loads['delay']))
print('*************************************************')
print('total time:\n',loads['total_time'])
save_plot = Path('./test7.jpg')
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(200, 80))
legend_labels = []
for cl in rs:
    #print(cl['W'])
    print(cl['accuracy'])
    print('max accuracy:   ',np.max(cl['accuracy']))
    legend_labels.append(cl['name'])
    ax_acc.plot(np.arange(50), cl['accuracy'], linewidth=7.0)
    ax_loss.plot(np.arange(50), cl['loss'], linewidth=7.0)

ax_acc.legend(legend_labels, )
#ax_acc.xaxis("Round")
#ax_acc.yaxis("Test Accuracy(%)")
ax_acc.set_title("Accuracy(%)")
fig.savefig(save_plot)
plt.clf()