
import matplotlib.pyplot as plt
import seaborn as sns


def draw_heatmap_k_m(file_name, file_c, alpha, nval):
    
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "times"
    plt.rcParams.update({'font.style': 'normal'})
    plt.rcParams.update({
        "text.usetex": True
    })

    
    # print(len(data[0]))
    core_data = [[0 for j in range(11)] for i in range(12)]
    jr_data = [[0 for j in range(11)] for i in range(12)]
    # cnt = 0
    # print(len(data))
    for f_num in range(len(file_name)):
        f_name = file_name[f_num]
        file_co = file_c[f_num]
        with open('{}.json'.format(f_name), 'r') as f:
            data = json.load(f)
        for i in range(len(data)):
            for j in range(len(data[i])):
                for k in range(len(data[i][j])):
                    # if k+2 == 5:
                    # print(j, k)
                    if all_n[i] == nval:
                        core_data[11-k][j] += data[i][j][k][0]/(file_co*len(file_name))
                        jr_data[11-k][j] += data[i][j][k][1]/(file_co*len(file_name))
                    if -1 == nval:
                        core_data[11-k][j] += data[i][j][k][0]/(file_co*len(file_name)*21)
                        jr_data[11-k][j] += data[i][j][k][1]/(file_co*len(file_name)*21)
                        # cnt += 1
                # core_data[i][j] /= len(data[i][j])
    
    fig, ax = plt.subplots(figsize=(4, 3.7), dpi=1000)
    plt.subplots_adjust(left=0, right=0.001, top=0.001, bottom = 0.00)
    # fig, ax = plt.subplots()
    sns.set(rc={'font.size': 20, 'font.family': 'times', 'font.style': 'normal', 'text.usetex' : True})
    fig.tight_layout()
    # heatmap = sns.heatmap(core_data, fmt=".1f", cmap="crest", xticklabels= all_m, yticklabels= all_n, square=True, ax=ax)

    if nval == -1:
        if alpha == 0:
            heatmap = sns.heatmap(jr_data, annot=False,vmin=0, vmax=0.23, fmt=".1f", cmap=sns.color_palette("Blues", as_cmap=True), xticklabels= all_m, yticklabels= all_k, square=True, ax=ax)
        else:
            heatmap = sns.heatmap(jr_data, annot=False,vmin=0, vmax=0.5, fmt=".1f", cmap=sns.color_palette("Blues", as_cmap=True), xticklabels= all_m, yticklabels= all_k, square=True, ax=ax)
    else:
        heatmap = sns.heatmap(jr_data, annot=False,vmin=0, vmax=2, fmt=".2f", cmap="crest", xticklabels= all_m, yticklabels= all_k, square=True, ax=ax)




    # Set the title and labels
    if nval == -1:
        heatmap.set_title(r'$\alpha$ = {}, Average over $n$'.format(alpha))
        heatmap.set_title(" ")
    else:
        heatmap.set_title(r'$\alpha$ = {}, $n = {}$'.format(alpha, nval))
    heatmap.set_ylabel("$b$")
    heatmap.set_xlabel("$m$")
    if nval == -1:
        if alpha == 0:
            plt.savefig('AverageHeatmap.pdf', dpi=1000)
        else:
            plt.savefig('PF_HeatMap_Average_alpha_{}.pdf'.format(alpha), dpi=1000)
    else:
        plt.savefig('PF_HeatMap_FixedN{}_alpha_{}.pdf'.format(nval, alpha), dpi=1000)
    plt.clf()



plt_labels = ['Alg. 1 not in Core', 'Alg. 1 not JR', 'Benchmark not in Core', 'Benchmark not JR']

def draw_alpha(file_names, alphas, nval):
    plt.style.use('default')
    # plt.style.use("seaborn-ticks")
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "times"
    plt.rcParams.update({'font.style': 'normal'})
    plt.rcParams.update({
        "text.usetex": True
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 5))
    fig.subplots_adjust(hspace=0.05)
    # plt.tight_layout()
    values = [[], [], [], []]
    print(len(file_names))
    for a in range(len(file_names)):
        temp = [[], [], [], []]
        cnt = 0
        for f_num in range(len(file_names[a])):
            with open('{}.json'.format(file_names[a][f_num][0]), 'r') as f:
                data = json.load(f)
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        for k in range(len(data[i][j])):
                            if all_n[i] == nval or nval == -1:
                                for x in range(4):
                                    # print(a, f_num)
                                    temp[x].append(data[i][j][k][x]/file_names[a][f_num][1])
                            # cnt += 1
        for x in range(4):
            values[x].append(temp[x])
    c = ['red', 'royalblue', 'darkorange', 'limegreen']
    for x in range(4):
        print(len(values[x][0]))
    avg_results = [[np.mean(x) for x in values[i]] for i in range(4)]
    sem_results = [[np.std(x)/np.sqrt(np.size(x)) for x in values[i]] for i in range(4)]


    for x in range(2, 4):
        ax1.errorbar(alphas, avg_results[x], label=plt_labels[x], yerr=sem_results[x], color=c[x])
    for x in range(2):
        ax2.errorbar(alphas, avg_results[x], label=plt_labels[x], yerr=sem_results[x], color=c[x])


    # zoom-in / limit the view to different portions of the data
    if nval == -1:
        ax2.set_ylim(-0.01, .12)  # outliers only
        ax1.set_ylim(5.1, 9)  # most of the data

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()


    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    ax2.set_xlabel(r'$\alpha$')
    
    if nval == -1:
        fig.suptitle(" ")
    else:
        fig.suptitle(" ")
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    
    fig.legend(lines, labels, loc='upper center', ncol=2, fancybox=True, bbox_to_anchor=(0.511, 0.99), facecolor='white', framealpha=0.95)

    # plt.legend()
    if nval == -1:
        if alpha == 0:
            plt.savefig( "AverageHeatmap.pdf", dpi=200 )
        else:
            plt.savefig( "FrequencyFairnessViolations.pdf", dpi=200 )
    else:
        plt.savefig( "FrequencyFixedN_{}.pdf".format(nval), dpi=200 )

    plt.show()
all_n = list(range(5, 26))
all_m = list(range(5, 16))
all_k = list(range(3, 15))

all_k.reverse()
all_n.reverse()

all_alpha = [1, 5, 9]


for ten_alpha in all_alpha:
    alpha = ten_alpha/10
    file_name = []
    # file_name = ['2alg_core_and_pf_{}'.format(ten_alpha), 'G3_2alg_{}'.format(ten_alpha)]

    for i in range(5):
        file_name.append('Exp_{}_{}_runs_{}'.format(i, ten_alpha, runs[i]))

    draw_heatmap_k_m(file_name, [ 0.5, 0.5, 1, 0.5, 0.5, 1], alpha, -1 )


all_alpha = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

file_names = []
alphas = []

runs = [100, 100, 100, 100]

for ten_alpha in all_alpha:
    alpha = ten_alpha/10
    if ten_alpha > 10:
        alpha = ten_alpha/100
    temp = []
    for i in range(len(runs)):
        temp.append(['Exp_{}_{}_runs_{}'.format(i, ten_alpha, runs[i]), runs[i]/100])
    file_names.append(temp)
    alphas.append(alpha)


all_nval = [-1, 5, 10, 15, 20, 25]
for nval in all_nval:
    draw_alpha(file_names, alphas, nval)
