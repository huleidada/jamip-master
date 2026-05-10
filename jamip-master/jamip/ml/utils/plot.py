import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def boxplot(data):
    plt.figure(figsize=(6,5),dpi=144)
    labels = list(data.columns)
    values = data.values.T.tolist()

    box = plt.boxplot(values,
        sym='v',
        showmeans=True,
        patch_artist=True,
        meanprops = {'marker':'o', 
                     'markersize':4, 
                     'markerfacecolor':'w', 
                     'markeredgecolor':'r'}, 
        medianprops = {'linestyle':'--', 'color':'k'},
        labels=labels
        )
    plt.grid(True)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    # colors = ['pink', 'lightblue', 'lightgreen']
    for patch,color in zip(box['boxes'],colors):
        patch.set_facecolor(color)
    plt.savefig('box.png')

def violinplot(data):
    import numpy as np
    plt.figure(figsize=(6,5),dpi=144)
    labels = list(data.columns)
    values = data.values.T.tolist()

    violin = plt.violinplot(values,
                 showmeans=False,
                 showmedians=False,
                 showextrema=False
                 )
    ax = plt.gca()
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('features')

    quartile1, medians, quartile3 = np.percentile(values, [25, 50, 75], axis=1)
    inds = np.arange(1, len(labels) + 1)
    ax.scatter(inds, medians, marker='o', color='w', s=20,
               edgecolor='r', zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', lw=3)
    for x,arr,q1,q3 in zip(inds,values,quartile1,quartile3):
        qmin = np.max([np.min(arr), q1-(q3-q1)*1.5])
        qmax = np.min([np.max(arr), q3+(q3-q1)*1.5])
        ax.plot([x,x],[qmin,qmax],color='k',lw=1.2)

    colors = list(mcolors.TABLEAU_COLORS.keys())
    for patch,color in zip(violin['bodies'],colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('k')
        patch.set_alpha(0.9)
    plt.savefig('violin.png')

def hist(data):
    plt.figure(figsize=(6,5),dpi=144)
    plt.hist(data.values, edgecolor='k',label=list(data.columns))
    plt.legend()
    plt.savefig('hist.png')
