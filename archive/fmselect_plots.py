import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}  \usepackage{amssymb}  \usepackage{mathrsfs}'
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import seaborn as sns


# path = "/Users/smaity/projects/routing/fmselect"

    
fmselect_load = np.load("IBMMIX_ID_plots_data.npy", allow_pickle=True)
data = fmselect_load.item()

prompts = data["prompts"]
categories = data["categories"]

models = list(data.keys())[2:]

model_names = ['claude-3-5-sonnet-v1',
 'titan-text-premier-v1',
 'gpt-4o',
 'gpt-4o-mini',
 'granite-3-2b-instruct',
 'granite-3-8b-instruct',
 'llama-3-1-70b-instruct',
 'llama-3-1-8b-instruct',
 'llama-3-2-1b-instruct',
 'llama-3-2-3b-instruct',
 'llama-3-3-70b-instruct',
 'llama-3-405b-instruct',
 'mixtral-8x7b-instruct-v01']

cost_true = []
cost_pred = []
perf_true = []
perf_pred = []


for m in models:
    cost_true.append(data[m]['actual cost'])
    cost_pred.append(data[m]['predicted cost'].numpy())
    perf_true.append(data[m]['actual perf'])
    perf_pred.append(data[m]['predicted perf'])

cost_pred = np.array(cost_pred).T
cost_true = np.array(cost_true).T
perf_pred = np.array(perf_pred).T
perf_true = np.array(perf_true).T
perf_pred = 1/(1 + np.exp(-perf_pred))

def route(scores, cost, cost_pred, correctness, lamb_range = np.arange(0, 1.001, 0.001)):
    router_cost = np.zeros(shape = (scores.shape[0], lamb_range.shape[0]))
    router_perf = np.zeros_like(router_cost)
    
    model_idx_all = np.zeros_like(router_cost)
    
    for idx_lam, lam in enumerate(lamb_range):
        model_idx = ((1 - lam) * scores - lam * cost_pred * 1000).argmax(axis = 1, keepdims = True)
        router_perf[:, idx_lam] = np.take_along_axis(correctness, model_idx, axis = 1).reshape((-1))
        router_cost[:, idx_lam] = np.take_along_axis(cost, model_idx, axis = 1).reshape((-1))
        model_idx_all[:, idx_lam] = model_idx[:, 0]

    return router_cost, router_perf, model_idx_all

router_cost, router_perf, model_idx_all = route(perf_pred, cost_true, cost_pred, perf_true)


cost_mean = cost_true.mean(axis = 0)
perf_mean = perf_true.mean(axis = 0)
router_cost_mean = router_cost.mean(axis = 0)
router_perf_mean = router_perf.mean(axis = 0)


#########################
# TRADEOFF CURVE

fig, ax = plt.subplots(1, 1, figsize = (4, 4))
markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'x', '+', 'h', 'H', 'd', '>']
for i, txt in enumerate(models):

    x, y = cost_mean[i], perf_mean[i]
    if models[i] not in []:
        ax.scatter([x], [y], marker = markers[i])#, label = models[i])
        if i == 200:
            ax.annotate(model_names[i], (x - 0.0012, y -0.03))
        elif i in [2, 3, ]:
            ax.annotate(model_names[i], (x + 0.0001, y))
        else:
            continue
    
ax.set_xlabel('cost (in \$)', fontsize = 15)
ax.set_ylabel('accuracy', fontsize = 15)
ax.plot(router_cost_mean, router_perf_mean, color = 'k', linestyle='--', linewidth=2, alpha = 0.75, label = "CARROT (Roberta)")
ax.legend(loc = "lower right")
ax.grid(True)
# ax.set_xscale("log")
# ax.set_xlim(1e-2, 25)
fig.savefig('plots/fmselect_perf_cost.pdf', bbox_inches = 'tight')



##########################
# MODEL COUNT


perfs = [0.6, 0.7, 0.8, 0.83, 0.85, 0.86]
idxs = [np.abs(router_perf_mean - p).argmin() for p in perfs]
perfs_true = [router_perf_mean[i] for i in idxs]
costs_true = [router_cost_mean[i] for i in idxs]
x_annot = ["(" + str((1000 * c).round(2)) + r"e-4, " + str(p.round(3)) + ")" for (c, p) in zip(costs_true, perfs_true)]

model_counts = np.zeros(shape = (len(idxs), len(models)))
for i_idx, idx in enumerate(idxs):
    for i_m in range(len(models)):
        model_counts[i_idx, i_m] = (model_idx_all[:, idx] == i_m).sum()

model_props = model_counts / model_idx_all.shape[0]

fig, ax = plt.subplots(1, 1, figsize = (3, 6))
sns.heatmap(model_props.T, yticklabels=model_names, annot=True, cbar=False, xticklabels=x_annot, fmt=".3f", ax=ax, cmap = "YlGn")
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
fig.savefig('plots/fmselect_model_selection.pdf', bbox_inches = 'tight')


#################################
# PLOT BY CATEGORY


markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'x', '+', 'h', 'H', 'd', '>']
colors = ['r', 'green','orange', 'k', 'b', 'y', 'purple']

# fig, ax = plt.subplots(1, 1, figsize = (10, 10))

dataset_cats = ["gpqa", "MuSR", 'TIGER-Lab/MMLU-Pro', 
                'lighteval/MATH',
                'openhermes/teknium',"ragbench"]
data_names = ["gpqa", "MuSR", 'MMLU-Pro', 'MATH',
                'openhermes',"ragbench"]


cost_ratios = []
perf_ratios = []


for i, (cat, cat_name) in enumerate(zip(dataset_cats, data_names)):
    idx = np.where(np.char.find(categories, cat)>=0)[0]
    
    cost_mean = cost_true[idx, :].mean(axis = 0)
    perf_mean = perf_true[idx, :].mean(axis = 0)
    router_cost_mean = router_cost[idx, :].mean(axis = 0)
    router_perf_mean = router_perf[idx, :].mean(axis = 0)
    
    gpt4_cost = cost_mean[2]
    gpt4_perf = perf_mean[2]
    
    cost_ratio = router_cost_mean/gpt4_cost
    perf_ratio = router_perf_mean/gpt4_perf
    
    cost_ratios.append(cost_ratio)
    perf_ratios.append(perf_ratio)
    
#     ax.plot(cost_ratio, perf_ratio, color = colors[i], linestyle='-', 
#             linewidth=1, alpha = 0.75, label = cat_name, marker = markers[i])
    
# ax.legend()


from math import pi


props = [0.1, 0.2, 0.3,]# 0.5]
props_name = ["10\% of gpt-4o cost",
              "20\% of gpt-4o cost",
              "30\% of gpt-4o cost",]
              # "50\% of gpt-4o cost",]
var = []
for p in props:
    var_d = []
    for d in range(len(dataset_cats)):
        cost_d = cost_ratios[d]
        perf_d = perf_ratios[d]
        var_d.append(perf_d[cost_d <= p].max().round(4))
        
    var.append(var_d)

var = np.array(var)


df_dir ={'group': props_name,}
for i, cat_name in enumerate(data_names):
    df_dir[cat_name] = var[:, i]
 
# Set data
df = pd.DataFrame(df_dir)

# ------- PART 1: Create background
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
# plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
# plt.ylim(0.7,1.1)

# Ind1
for i in range(len(props)):
    values=df.loc[i].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=props_name[i], color = colors[i])
    ax.fill(angles, values, color = colors[i], alpha=0.05)

N_0 = 1000    
angles_0 = [n / float(N_0) * 2 * pi for n in range(N_0)]

ax.plot(angles_0, np.ones_like(angles_0), linewidth=1.5, linestyle='--', color = "blue")
ax.tick_params(axis='both', which='major', labelsize=12)
 
# Add legend
fig.legend(loc='upper right')
# Show the graph
# plt.show()    

fig.savefig('plots/fmselect_gpt4o_comparison.pdf', bbox_inches = 'tight')

