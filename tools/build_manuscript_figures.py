"""Draw publication figures directly from compact saved numerical evidence."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'publication/evidence'
DESTINATION = ROOT / 'publication/figures'
COLORS = {'original': '#23658b', 'control': '#428165', 'residual': '#815f95', 'gray': '#626b72'}
LABELS = {'raw_esm_zero_shot': 'Sequence only', 'esm_conservation_logistic': 'Sequence + conservation',
          'lightgbm': 'Original LightGBM', 'concatenation': 'Concatenation', 'gated_fusion': 'Gated fusion',
          'cross_attention': 'Cross attention', 'reliability_residual': 'Reliability residual',
          'evidential_residual': 'Evidential residual', 'reliability_control': 'Reliability control',
          'fixed_quality_gate': 'Fixed quality gate', 'hard_switch': 'Hard switch',
          'no_modality_dropout': 'No modality dropout', 'unbounded_residual': 'Unbounded residual',
          'same_input_mlp': 'Shared input MLP', 'same_input_logistic': 'Shared input logistic',
          'same_input_lightgbm': 'Shared input LightGBM'}


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8'))


def save(fig, name):
    DESTINATION.mkdir(parents=True, exist_ok=True)
    fig.savefig(DESTINATION / (name + '.png'), dpi=350, facecolor='white', bbox_inches='tight')
    matplotlib.rcParams['svg.hashsalt'] = 'varifuse-recorded-evidence-20261004'
    svg = DESTINATION / (name + '.svg')
    fig.savefig(svg, facecolor='white', bbox_inches='tight', metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())
                   + '\n', encoding='utf-8', newline='\n')
    plt.close(fig)


def box(ax, xy, size, title, lines, color):
    x, y = xy
    width, height = size
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.01,rounding_size=0.015',
                               linewidth=1.1, edgecolor=color, facecolor='#f6f8fa'))
    ax.text(x + width / 2, y + height - .038, title, fontsize=10, weight='bold',
            ha='center', va='top', color=color)
    body_offset = .033 if len(lines) >= 4 else .013
    ax.text(x + width / 2, y + height / 2 - body_offset, '\n'.join(lines), fontsize=9,
            ha='center', va='center', linespacing=1.5)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12,
                                linewidth=1.1, color='#646c73'))


def figure_one():
    fig, ax = plt.subplots(figsize=(7.4, 7.3))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    box(ax, (.04, .76), (.42, .21), 'Development cohort',
        ['ClinVar June 2024 labels', '48,197 variants | 16,740 positives', '4,692 genes | 3,718 split groups'], COLORS['original'])
    box(ax, (.54, .76), (.42, .21), 'Later clinical cohort',
        ['ClinVar August 2026', '14,650 variants | 7,323 positives', '3,059 genes | exact variants excluded'], COLORS['original'])
    box(ax, (.04, .48), (.42, .21), 'Original nested comparison',
        ['5 outer folds x 3 inner folds', 'Fit, stop, calibration and threshold', 'partitions separate from test groups'], COLORS['original'])
    box(ax, (.54, .48), (.42, .21), 'Clinical transfer',
        ['92.01% structure coverage', '98.54% conservation coverage', '679 variants in gene-disjoint subset'], COLORS['original'])
    arrow(ax, (.25, .76), (.25, .70))
    arrow(ax, (.75, .76), (.75, .70))
    box(ax, (.04, .22), (.42, .20), 'Later exploratory controls',
        ['8 models x 5 frozen folds', 'Shared scalar and ESM inputs', 'Clinical cohort already inspected'], COLORS['control'])
    box(ax, (.54, .22), (.42, .20), 'Separate functional endpoint',
        ['696,311 single substitutions', '217 assays | 186 proteins',
         'No main-cohort structure', 'or conservation annotations'], COLORS['residual'])
    arrow(ax, (.25, .48), (.25, .43))
    ax.text(.5, .12, 'Covariates: dbNSFP 5.3a | UniProt January 2026 | AlphaFold v6',
            ha='center', fontsize=9)
    ax.text(.5, .066, 'Label-temporal evaluation does not reconstruct all historical covariates.',
            ha='center', fontsize=9, color='#444444')
    save(fig, 'figure_1_cohorts')


def figure_two():
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    box(ax, (.04, .76), (.40, .21), 'Sequence score s',
        ['Masked alternate minus reference', 'log probability from frozen ESM2'], COLORS['original'])
    box(ax, (.56, .76), (.40, .21), 'Monotonic anchor',
        ['b - softplus(a) s', 'Positive slope on deleteriousness'], COLORS['original'])
    arrow(ax, (.45, .86), (.55, .86))
    box(ax, (.04, .47), (.40, .21), 'Residual inputs',
        ['Masked residue vector and context', 'Local structure and conservation', 'Encoded into a bounded correction'], COLORS['residual'])
    box(ax, (.56, .46), (.40, .23), 'Quality and availability gate',
        ['g = availability x quality', 'x learned gate',
         'g = 0 if both auxiliary sources', 'are unusable'], COLORS['control'])
    box(ax, (.18, .18), (.64, .20), 'Fused logit',
        ['z = anchor + g c tanh(h)', 'When g = 0, the model returns its anchor.',
         'A separate calibrator and threshold still apply.'], COLORS['gray'])
    ax.plot([.97, .985, .985], [.86, .86, .28], color='#646c73', linewidth=1.1)
    arrow(ax, (.985, .28), (.84, .28))
    arrow(ax, (.25, .46), (.38, .39))
    arrow(ax, (.75, .46), (.62, .39))
    ax.text(.5, .079, 'Controls remove the learned gate, continuity, dropout or residual bound.',
            fontsize=9, ha='center')
    save(fig, 'figure_2_model')


def dotplot(ax, models, values, title, color):
    y = np.arange(len(models))
    ax.scatter(values, y, s=37, c=color, zorder=3)
    ax.set_yticks(y, [LABELS[m] for m in models], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, loc='left', weight='bold', pad=12)
    ax.set_xlabel('Internal average precision', fontsize=9)
    ax.set_xlim(.83, .945)
    ax.grid(axis='x', alpha=.20)
    for yy, value in zip(y, values):
        ax.annotate(f'{value:.4f}', (value, yy), xytext=(5, 0), textcoords='offset points',
                    fontsize=8, va='center')
    ax.spines[['top', 'right']].set_visible(False)


def figure_three():
    original = read('original_verified_evidence.json')['metrics']['internal']
    controls = read('control_metrics.json')['internal']
    intervals = read('control_internal_intervals.json')
    fig = plt.figure(figsize=(7.6, 7.7), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    ax1, ax2 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    models = ['raw_esm_zero_shot', 'esm_conservation_logistic', 'lightgbm', 'concatenation',
              'cross_attention', 'gated_fusion', 'reliability_residual', 'evidential_residual']
    dotplot(ax1, models, [original[m]['auprc'] for m in models], 'A  Original input contracts', COLORS['original'])
    models = list(controls)
    dotplot(ax2, models, [controls[m]['auprc'] for m in models], 'B  Exploratory shared inputs', COLORS['control'])
    ax3 = fig.add_subplot(grid[1, :])
    comparisons = ['same_input_lightgbm', 'same_input_mlp', 'fixed_quality_gate', 'unbounded_residual']
    for row, name in enumerate(comparisons):
        metric = intervals[name + '_minus_reliability_control']['metrics']['auprc']
        low, high = metric['ci95']
        value = metric['difference']
        ax3.errorbar(value, row, xerr=np.array([[value-low], [high-value]]), fmt='o',
                     color=COLORS['control'], capsize=3)
        ax3.annotate(f'{value:+.4f} [{low:+.4f}, {high:+.4f}]', (high, row),
                     xytext=(6, 0), textcoords='offset points', fontsize=8, va='center')
    ax3.axvline(0, linestyle='--', color='#777777', linewidth=.9)
    ax3.set_yticks(range(4), [LABELS[x] for x in comparisons], fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlim(-.003, .048)
    ax3.set_xlabel('AP difference relative to reliability control', fontsize=9)
    ax3.set_title('C  Paired connected-group intervals conditional on saved predictions',
                  fontsize=10, loc='left', weight='bold', pad=12)
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.grid(axis='x', alpha=.20)
    save(fig, 'figure_3_comparison')


def figure_four():
    original = read('original_verified_evidence.json')
    clinical = original['metrics']['clinvar_exact_variant_disjoint']
    masked = original['full_mask_clinvar']
    dms = read('dms_aggregation.json')['spearman']
    models = ['raw_esm_zero_shot', 'reliability_residual', 'gated_fusion', 'cross_attention']
    display = ['Sequence', 'Residual', 'Gated', 'Attention']
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.5), layout='constrained')
    x = np.arange(4)
    for ax, metric, title, ylabel in [
            (axes[0, 0], 'auprc', 'A  Clinical ranking', 'Average precision'),
            (axes[1, 0], 'brier', 'C  Clinical probability error', 'Brier score (lower is better)'),
            (axes[1, 1], 'mcc', 'D  Clinical decisions', 'Matthews correlation coefficient')]:
        if metric == 'auprc':
            for i, model in enumerate(models):
                ax.plot([i - .14, i + .14], [clinical[model][metric], masked[model][metric]],
                        color='#aaaaaa', linewidth=1, zorder=1)
            ax.scatter(x - .14, [clinical[m][metric] for m in models], s=36,
                       color=COLORS['original'], label='Observed annotations', zorder=3)
            ax.scatter(x + .14, [masked[m][metric] for m in models], s=36, marker='s',
                       color=COLORS['residual'], label='Joint auxiliary mask', zorder=3)
        else:
            ax.bar(x - .18, [clinical[m][metric] for m in models], .35,
                   color=COLORS['original'], label='Observed annotations')
            ax.bar(x + .18, [masked[m][metric] for m in models], .35,
                   color=COLORS['residual'], label='Joint auxiliary mask')
        ax.set_xticks(x, display, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, weight='bold', loc='left')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
    axes[0, 0].set_ylim(.85, .98)
    axes[0, 0].legend(fontsize=7, loc='lower left', frameon=False)
    ax = axes[0, 1]
    names = ['raw_esm_zero_shot', 'reliability_residual', 'gated_fusion']
    x = np.arange(3)
    ax.bar(x - .18, [dms[m]['assay_macro'] for m in names], .35,
           color=COLORS['original'], label='Assay macro')
    ax.bar(x + .18, [dms[m]['score'] for m in names], .35,
           color=COLORS['control'], label='Protein/category summary')
    ax.set_xticks(x, ['Sequence', 'Residual', 'Gated'], fontsize=8)
    ax.set_ylim(0, .5)
    ax.set_ylabel('Functional Spearman correlation', fontsize=9)
    ax.set_title('B  DMS sequence fallback', fontsize=10, weight='bold', loc='left')
    ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    save(fig, 'figure_4_fallback')


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
    figure_one()
    figure_two()
    figure_three()
    figure_four()
    print('Created four data-bound PNG and SVG figures:', DESTINATION)


if __name__ == '__main__':
    main()
