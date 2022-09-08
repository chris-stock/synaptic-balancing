"""
Plotting code for figures.
"""

from matplotlib import pyplot as plt
import numpy as np

yellow_rgb = np.array([251, 176, 59]) / 256.
purple_rgb = np.array([102, 45, 145]) / 256.
red_rgb = np.array([193, 39, 45]) / 256.
teal_rgb = np.array([0, 169, 157]) / 256.
light_purple_rgb = tuple(1 - .6 * (1 - x) for x in purple_rgb)

DPI = 300


def plot_network_balance_during_training(
    fig_path,
    g_norm,
    gf_true,
    gf_shuff_rows,
    l2_reg_scale
):

    figsize = (3, 2.5)
    fontsize = 8
    legendfontsize = 6
    xticks = [0, 2000, 4000, 6000]
    yticks = [0, .5, 1, 1.5, 2, 2.5]
    linewidth = 1
    line_colors = [yellow_rgb, red_rgb, purple_rgb]
    niter = 6000
    extra_x = 500
    hist_scale = 30

    fig, ax = plt.subplots(1, figsize=figsize)

    for gn, c in zip(g_norm, line_colors):
        ax.plot(gn, c=c, linewidth=linewidth)
    for i, gf in enumerate(gf_shuff_rows):
        counts, bins = np.histogram(gf_shuff_rows[i], density=True)
        ax.hist(
            bins[:-1],
            bins,
            weights=hist_scale * counts,
            color=[1 - .7 * (1 - v) for v in line_colors[i]],
            orientation='horizontal',
            bottom=niter,
        )
        ax.plot(
            [niter, niter + extra_x],
            [gf_true[i], gf_true[i]],
            lw=linewidth,
            c=line_colors[i],
            linestyle=':'
        )

    ax.set_title('Network balance during training \n '
                 'with $\ell_2$ regularization ($\lambda$)', fontsize=fontsize
                 )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    # ax.set_ylabel('Norm of neural gradient', fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=fontsize)

    ax.set_xlabel('Training iteration', fontsize=fontsize)
    ax.set_ylabel('Norm of neural gradients', fontsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=fontsize)

    legend_labels = ['$\lambda = {}$'.format(zr) for zr in l2_reg_scale]
    legend_labels[-1] = legend_labels[-1] + ' ' * 10
    ax.legend(
        legend_labels,
        fontsize=legendfontsize,
        loc='upper left',
        bbox_to_anchor=(.55, 0, .1, .75),
        borderaxespad=0,
        frameon=False,
        ncol=1,
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=DPI)


def plot_task_loss_with_neural_noise(
    fig_path,
    normalized_noise_levels,
    losses_orig,
    losses_balanced
):


    m = 13
    figsize=(3,2.7)
    fontsize=8
    legendfontsize=6
    linewidth=1
    yticks= np.arange(50., step=10.)
    xticks = np.arange(.5, step=.1)
    xticklabels = ['{:.1f}'.format(v) for v in xticks]
    title='Loss on task as function of injected noise\n ' \
          'before and after balancing'
    # xlim = (0,3)
    # ylim = (0, 100)
    # xlim = (0,2)
    # ylim = (0, 60)


    fig, ax = plt.subplots(
        1,
        figsize=figsize,
    )

    ax.plot(
        normalized_noise_levels[:m],
        losses_orig[:m],
        c=purple_rgb,
        linestyle='-',
        lw=linewidth,
    )
    ax.plot(
        normalized_noise_levels[:m],
        losses_balanced[:m],
        c=light_purple_rgb,
        linestyle='-',
        lw=linewidth,
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    # ax.set_xlim(xlim)
    ax.set_xlabel('Std. dev. of injected noise \n '
                  '(fraction of std. dev. of firing rates)',
                  fontsize=fontsize)
    ax.set_ylabel('Loss', fontsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5.))
    ax.set_xticklabels(xticklabels, fontsize=fontsize)
    ax.set_yticklabels(yticks.astype(int), fontsize=fontsize)

    ax.legend(
        ['original network', 'balanced network'],
        fontsize=legendfontsize,
        frameon=False,
        loc='lower right',
        ncol=1,
        borderaxespad=0,
    )

    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=DPI)
    plt.show()


def plot_inset_total_cost_bars(
    fig_path,
    balancing_results
):

    figsize=(1,.9)
    xlim=(0,3)
    xticks= []
    xticklabels=['', '']
    yticks= [0,2,4,6]
    fontsize=7

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.bar(
        [1, 2],
        [balancing_results['c0'], balancing_results['cf']],
        color=[purple_rgb, light_purple_rgb]
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_xlim(xlim)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=fontsize)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1.))
    ax.set_ylabel('Total cost $C$', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=DPI)