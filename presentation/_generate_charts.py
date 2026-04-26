"""
Regenerates all presentation charts as PNGs in presentation/charts/.

Numbers are hardcoded from our actual model runs (matches JOURNEY.md and the
v5_external_features snapshot). Run once after any change.

    python presentation/_generate_charts.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

OUT = Path(__file__).parent / "charts"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 130,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
})

C_GOOD = '#2ca02c'
C_BAD  = '#d62728'
C_NEU  = '#7f7f7f'
C_HI   = '#1f77b4'
C_HI2  = '#ff7f0e'


# ---------------------------------------------------------------------------
# Chart 1: Feature engineering phases (v1, v2, v3, v5, current — v4 is a
# model-only change and is intentionally omitted)
# ---------------------------------------------------------------------------
phases = pd.DataFrame([
    {'phase': 'v1\n240 raw lag\nfeatures',                    'rmse': 9.811, 'kept': True},
    {'phase': 'v2\nEngineered\nfeature sets',                 'rmse': 9.846, 'kept': False},
    {'phase': 'v3\nPer-36 rate\nnormalization',               'rmse': 10.147,'kept': False},
    {'phase': 'v5\nExternal + derived\nfeatures',             'rmse': 9.527, 'kept': True},
    {'phase': 'current\nPruned\nfeature set',                 'rmse': 9.533, 'kept': True},
])

fig, ax = plt.subplots(figsize=(11, 5.5))
x = range(len(phases))
colors = [C_GOOD if k else C_BAD for k in phases['kept']]
bars = ax.bar(x, phases['rmse'], color=colors, edgecolor='black', linewidth=0.6, alpha=0.85)

ax.axhline(9.811, color=C_NEU, linestyle='--', linewidth=1, alpha=0.6)
ax.text(len(phases)-0.5, 9.815, 'v1 baseline', color=C_NEU, fontsize=9, ha='right', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(phases['phase'], fontsize=10)
ax.set_ylabel('Test RMSE (fantasy points)')
ax.set_title('Test RMSE across feature engineering phases')
ax.set_ylim(9.3, 10.3)

for bar, val in zip(bars, phases['rmse']):
    ax.text(bar.get_x()+bar.get_width()/2, val + 0.02, f'{val:.3f}',
            ha='center', fontsize=10, fontweight='bold')

ax.legend(handles=[Patch(facecolor=C_GOOD, label='Feature change improved test RMSE'),
                   Patch(facecolor=C_BAD,  label='Feature change did not help / regressed')],
          loc='upper right', fontsize=9)

ax.text(0.99, -0.20,
        'v4 (model architecture experiment) intentionally omitted — see model writeup',
        transform=ax.transAxes, ha='right', fontsize=8, color=C_NEU, style='italic')

plt.tight_layout()
plt.savefig(OUT / "01_journey.png")
plt.close()


# ---------------------------------------------------------------------------
# Chart 2: Feature group ablation
# ---------------------------------------------------------------------------
ablation = pd.DataFrame([
    {'config': 'rolling only (75 feats)',     'rmse': 9.839, 'delta': 0.000},
    {'config': '+ game context (3)',          'rmse': 9.838, 'delta': -0.001},
    {'config': '+ trend L3-L10 (8)',          'rmse': 9.833, 'delta': -0.005},
    {'config': '+ efficiency rolling (9)',    'rmse': 9.812, 'delta': -0.024},
    {'config': '+ MISSING TEAMMATES (3)',     'rmse': 9.567, 'delta': -0.249},
    {'config': '+ schedule density (3)',      'rmse': 9.568, 'delta': +0.002},
    {'config': '+ player position (5)',       'rmse': 9.557, 'delta': -0.011},
    {'config': '+ DvP (1)',                   'rmse': 9.542, 'delta': -0.015},
])

fig, ax = plt.subplots(figsize=(11.5, 5.5))
y = range(len(ablation))
colors = [C_GOOD if d < -0.005 else (C_BAD if d > 0 else C_NEU) for d in ablation['delta']]
bars = ax.barh(y, ablation['delta'], color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(ablation['config'])
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Test RMSE delta when this group is added (negative = improvement)')
ax.set_title('Feature group ablation — Missing Teammates dominates')

for i, (bar, d, te) in enumerate(zip(bars, ablation['delta'], ablation['rmse'])):
    label = f'{d:+.3f}  →  RMSE {te:.3f}'
    x_pos = d + (-0.008 if d < 0 else 0.005)
    ha = 'right' if d < 0 else 'left'
    ax.text(x_pos, i, label, va='center', ha=ha, fontsize=9,
            fontweight='bold' if abs(d) > 0.05 else 'normal')

ax.set_xlim(-0.30, 0.05)
plt.tight_layout()
plt.savefig(OUT / "02_ablation.png")
plt.close()


# ---------------------------------------------------------------------------
# Chart 3: Final feature set composition
# ---------------------------------------------------------------------------
final_features = pd.DataFrame([
    {'group': 'Player rolling (L3/L5/L10)',   'count': 27, 'highlight': False},
    {'group': 'Team rolling (L3/L5/L10)',     'count': 24, 'highlight': False},
    {'group': 'Opponent rolling (L3/L5/L10)', 'count': 24, 'highlight': False},
    {'group': 'Player efficiency rolling',    'count': 9,  'highlight': False},
    {'group': 'Player trend (L3 - L10)',      'count': 8,  'highlight': False},
    {'group': 'Position one-hot + height/exp','count': 5,  'highlight': False},
    {'group': 'Game context',                 'count': 3,  'highlight': False},
    {'group': 'Schedule density',             'count': 3,  'highlight': False},
    {'group': 'Missing teammates (DNP)',      'count': 3,  'highlight': True},
    {'group': 'Defense vs Position (DvP)',    'count': 1,  'highlight': False},
])

fig, ax = plt.subplots(figsize=(11, 5.5))
y = range(len(final_features))
colors = [C_GOOD if h else C_HI for h in final_features['highlight']]
bars = ax.barh(y, final_features['count'], color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(final_features['group'])
ax.invert_yaxis()
ax.set_xlabel('Number of features in this group')
ax.set_title(f'Final feature set: {final_features["count"].sum()} features in 10 groups')
for bar, c in zip(bars, final_features['count']):
    ax.text(bar.get_width() + 0.3, bar.get_y()+bar.get_height()/2, str(c), va='center', fontsize=10)

ax.text(28, 8, 'Highest impact group:\n3 features → -0.249 RMSE',
        fontsize=10, color=C_GOOD, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor=C_GOOD, alpha=0.9))

plt.tight_layout()
plt.savefig(OUT / "03_final_feature_set.png")
plt.close()


# ---------------------------------------------------------------------------
# Chart 4: Cumulative test RMSE as features are added
# ---------------------------------------------------------------------------
cum = pd.DataFrame([
    {'step': 'rolling',          'rmse': 9.839},
    {'step': '+context',         'rmse': 9.838},
    {'step': '+trend',           'rmse': 9.833},
    {'step': '+efficiency',      'rmse': 9.812},
    {'step': '+missing\nteammates','rmse': 9.567},
    {'step': '+schedule',        'rmse': 9.568},
    {'step': '+position',        'rmse': 9.557},
    {'step': '+DvP',             'rmse': 9.542},
])

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(cum['step'], cum['rmse'], marker='o', markersize=10, color=C_HI, linewidth=2)

i_mt = 4
ax.annotate('THE BREAKTHROUGH\nmissing teammates: -0.249 RMSE',
            xy=(i_mt, cum['rmse'].iloc[i_mt]),
            xytext=(i_mt + 0.3, 9.72),
            fontsize=11, fontweight='bold', color=C_GOOD,
            arrowprops=dict(arrowstyle='->', color=C_GOOD, lw=2))

ax.annotate(f'{cum["rmse"].iloc[0]:.3f}', xy=(0, cum['rmse'].iloc[0]),
            xytext=(0, cum['rmse'].iloc[0]+0.02), ha='center', fontsize=10, fontweight='bold')
ax.annotate(f'{cum["rmse"].iloc[-1]:.3f}', xy=(len(cum)-1, cum['rmse'].iloc[-1]),
            xytext=(len(cum)-1, cum['rmse'].iloc[-1]-0.04), ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Test RMSE')
ax.set_title('Test RMSE as feature groups are added one at a time')
ax.set_ylim(9.45, 9.92)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "04_cumulative_rmse.png")
plt.close()


# ---------------------------------------------------------------------------
# Chart 5: Feature experiments only (model experiments excluded — they live
# in the model writeup)
# ---------------------------------------------------------------------------
experiments = pd.DataFrame([
    {'experiment': 'Missing teammates',          'delta': -0.249, 'kept': True},
    {'experiment': 'Player efficiency rolling',  'delta': -0.018, 'kept': True},
    {'experiment': 'Defense vs Position (DvP)',  'delta': -0.015, 'kept': True},
    {'experiment': 'Player position one-hot',    'delta': -0.011, 'kept': True},
    {'experiment': 'Vegas spread/total',         'delta': -0.009, 'kept': False},
    {'experiment': 'Game context (home/rest)',   'delta': -0.001, 'kept': False},
    {'experiment': 'Schedule density',           'delta': +0.002, 'kept': False},
    {'experiment': 'Trend (L3 - L10)',           'delta': +0.003, 'kept': False},
    {'experiment': 'Per-36 rate normalization',  'delta': +0.336, 'kept': False},
])

experiments = experiments.sort_values('delta').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11.5, 5.5))
y = range(len(experiments))
colors = [C_GOOD if k else (C_BAD if d > 0 else C_NEU) for k, d in zip(experiments['kept'], experiments['delta'])]
bars = ax.barh(y, experiments['delta'], color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(experiments['experiment'])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Test RMSE delta')
ax.set_title('Feature experiments — what we tried, sorted by impact')

for i, (bar, d) in enumerate(zip(bars, experiments['delta'])):
    label = f'{d:+.3f}'
    if d < 0:
        ax.text(d - 0.005, i, label, va='center', ha='right', fontsize=9,
                fontweight='bold' if d < -0.05 else 'normal')
    else:
        ax.text(d + 0.005, i, label, va='center', ha='left', fontsize=9)

ax.set_xlim(-0.30, 0.40)

ax.legend(handles=[Patch(facecolor=C_GOOD, label='Kept in final feature set'),
                   Patch(facecolor=C_NEU,  label='Dropped (small/no benefit)'),
                   Patch(facecolor=C_BAD,  label='Regressed (worse)')],
          loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "05_feature_experiments.png")
plt.close()


# ===========================================================================
# Model writeup charts (model_*.png)
# ===========================================================================

# ---------------------------------------------------------------------------
# Model chart 1: Test RMSE comparison across every model we tried
# ---------------------------------------------------------------------------
models = pd.DataFrame([
    {'model': 'LinearRegression\n(v1, 240 raw lags)',          'rmse': 9.811, 'final': False},
    {'model': 'RandomForest\n(v4, 75 rolling)',                'rmse': 9.942, 'final': False},
    {'model': 'HistGB default\n(v5, 107 features)',            'rmse': 9.542, 'final': False},
    {'model': 'HistGB tuned\n(RandomizedSearchCV)',            'rmse': 9.538, 'final': False},
    {'model': 'Per-position HistGB\n(separate G/F/C)',         'rmse': 9.553, 'final': False},
    {'model': 'Multi-output HistGB\n(7 models combined)',      'rmse': 9.527, 'final': True},
])

fig, ax = plt.subplots(figsize=(13, 5.5))
x = range(len(models))
colors = [C_GOOD if f else C_HI for f in models['final']]
bars = ax.bar(x, models['rmse'], color=colors, edgecolor='black', linewidth=0.6, alpha=0.85)

ax.axhline(9.811, color=C_NEU, linestyle='--', linewidth=1, alpha=0.6)
ax.text(len(models)-0.3, 9.815, 'baseline', color=C_NEU, fontsize=9, ha='right', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(models['model'], fontsize=9)
ax.set_ylabel('Test RMSE (fantasy points)')
ax.set_title('Test RMSE across every model we tried')
ax.set_ylim(9.3, 10.1)

for bar, val in zip(bars, models['rmse']):
    ax.text(bar.get_x()+bar.get_width()/2, val + 0.015, f'{val:.3f}',
            ha='center', fontsize=10, fontweight='bold')

ax.legend(handles=[Patch(facecolor=C_GOOD, label='Final production model'),
                   Patch(facecolor=C_HI,   label='Tested')],
          loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "model_01_comparison.png")
plt.close()


# ---------------------------------------------------------------------------
# Model chart 2: Train vs Test RMSE — the overfitting story
# ---------------------------------------------------------------------------
gap = pd.DataFrame([
    {'model': 'LinearRegression\n(v1)',        'train': 9.427, 'test': 9.811},
    {'model': 'RandomForest\n(v4)',            'train': 3.541, 'test': 9.942},
    {'model': 'HistGB default\n(v5)',          'train': 9.046, 'test': 9.542},
    {'model': 'HistGB tuned\n(v5)',            'train': 9.070, 'test': 9.538},
])

fig, ax = plt.subplots(figsize=(11, 5.5))
x = range(len(gap))
width = 0.38
ax.bar([i - width/2 for i in x], gap['train'], width, label='Train RMSE',
       color=C_HI, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.bar([i + width/2 for i in x], gap['test'], width, label='Test RMSE',
       color=C_HI2, edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(gap['model'], fontsize=10)
ax.set_ylabel('RMSE (fantasy points)')
ax.set_title('Train vs Test RMSE — RandomForest is the only one that overfit')
ax.set_ylim(0, 11)
ax.legend(loc='upper right', fontsize=10)

for i, (t, te) in enumerate(zip(gap['train'], gap['test'])):
    ax.text(i - width/2, t + 0.15, f'{t:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + width/2, te + 0.15, f'{te:.2f}', ha='center', fontsize=9, fontweight='bold')
    gap_val = te - t
    ax.text(i, max(t, te) + 0.7, f'gap: {gap_val:+.2f}',
            ha='center', fontsize=9, color=(C_BAD if gap_val > 1.5 else C_NEU),
            fontweight='bold')

plt.tight_layout()
plt.savefig(OUT / "model_02_train_vs_test.png")
plt.close()


# ---------------------------------------------------------------------------
# Model chart 3: Multi-output per-stat decomposition
# ---------------------------------------------------------------------------
decomp = pd.DataFrame([
    {'stat': 'PTS',  'weight': 1.00, 'rmse': 6.095},
    {'stat': 'REB',  'weight': 1.25, 'rmse': 2.539},
    {'stat': 'AST',  'weight': 1.50, 'rmse': 1.910},
    {'stat': 'STL',  'weight': 2.00, 'rmse': 0.969},
    {'stat': 'BLK',  'weight': 2.00, 'rmse': 0.783},
    {'stat': 'TOV',  'weight': -0.50,'rmse': 1.214},
    {'stat': 'FG3M', 'weight': 0.50, 'rmse': 1.306},
])
decomp['weighted_rmse'] = decomp['rmse'] * decomp['weight'].abs()
decomp['contribution_pct'] = decomp['weighted_rmse']**2 / (decomp['weighted_rmse']**2).sum() * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.bar(decomp['stat'], decomp['rmse'], color=C_HI, alpha=0.85, edgecolor='black', linewidth=0.5)
for i, (s, r) in enumerate(zip(decomp['stat'], decomp['rmse'])):
    ax.text(i, r + 0.1, f'{r:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Per-stat test RMSE')
ax.set_title('Per-stat prediction error\n(raw — before DK weighting)')
ax.set_ylim(0, 7.2)

ax = axes[1]
colors_c = [C_HI if c < 50 else C_HI2 for c in decomp['contribution_pct']]
ax.bar(decomp['stat'], decomp['contribution_pct'], color=colors_c, alpha=0.85, edgecolor='black', linewidth=0.5)
for i, (s, c) in enumerate(zip(decomp['stat'], decomp['contribution_pct'])):
    ax.text(i, c + 1, f'{c:.0f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('% of total weighted error variance')
ax.set_title('PTS dominates total error\n(after DK weighting)')
ax.set_ylim(0, max(decomp['contribution_pct']) * 1.15)

plt.tight_layout()
plt.savefig(OUT / "model_03_per_stat.png")
plt.close()


# ---------------------------------------------------------------------------
# Model chart 4: Per-position separate models — why they failed
# ---------------------------------------------------------------------------
positions = pd.DataFrame([
    {'pos': 'Guards (G)\n235k train',     'rmse': 9.582, 'n_train': 235937},
    {'pos': 'Forwards (F)\n137k train',   'rmse': 9.224, 'n_train': 137642},
    {'pos': 'Centers (C)\n119k train',    'rmse': 9.925, 'n_train': 119820},
    {'pos': 'Combined\n(per-position)',   'rmse': 9.553, 'n_train': 0},
    {'pos': 'Global model\n(no split)',   'rmse': 9.538, 'n_train': 0},
])

fig, ax = plt.subplots(figsize=(11, 5))
colors = [C_HI, C_HI, C_HI, C_BAD, C_GOOD]
bars = ax.bar(positions['pos'], positions['rmse'], color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)

ax.axhline(9.538, color=C_GOOD, linestyle='--', linewidth=1, alpha=0.6)
ax.text(0, 9.55, 'global model wins', color=C_GOOD, fontsize=9, va='bottom')

for bar, val in zip(bars, positions['rmse']):
    ax.text(bar.get_x()+bar.get_width()/2, val + 0.03, f'{val:.3f}',
            ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Test RMSE')
ax.set_title('Per-position separate models — combined regressed (+0.015) vs global')
ax.set_ylim(9.0, 10.1)
plt.tight_layout()
plt.savefig(OUT / "model_04_per_position.png")
plt.close()


# ===========================================================================
# Implications writeup charts (impl_*.png)
# ===========================================================================

# ---------------------------------------------------------------------------
# Implications chart 1: The predictability ladder
# ---------------------------------------------------------------------------
NAIVE_RMSE = 13.877
ladder = pd.DataFrame([
    {'level': 'Predict the mean\n(no model)',                   'rmse': 13.877, 'category': 'naive'},
    {'level': 'LinearRegression\n(our v1 baseline)',            'rmse': 9.811,  'category': 'ours'},
    {'level': 'Our final model\n(multi-output HistGB)',         'rmse': 9.527,  'category': 'ours_best'},
    {'level': 'Academic best\n(Stanford CS stats class)',       'rmse': 8.54,   'category': 'external'},
    {'level': 'Estimated ceiling\nwith real-time data',         'rmse': 7.5,    'category': 'ceiling'},
    {'level': 'Perfect prediction\n(impossible)',               'rmse': 0,      'category': 'perfect'},
])
ladder['r2'] = 1 - (ladder['rmse'] / NAIVE_RMSE) ** 2
ladder.loc[ladder['rmse'] == 0, 'r2'] = 1.0  # explicit for perfect

cat_colors = {
    'naive': C_NEU,
    'ours': C_HI,
    'ours_best': C_GOOD,
    'external': C_HI2,
    'ceiling': '#9467bd',
    'perfect': '#cccccc',
}

fig, ax = plt.subplots(figsize=(12.5, 6.5))
y = range(len(ladder))
colors = [cat_colors[c] for c in ladder['category']]
bars = ax.barh(y, ladder['rmse'], color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(ladder['level'])
ax.invert_yaxis()
ax.set_xlabel('Test RMSE (fantasy points)')
ax.set_title('The predictability ladder — where our model lands')
ax.set_xlim(0, 16)

for i, (bar, rmse, r2) in enumerate(zip(bars, ladder['rmse'], ladder['r2'])):
    if rmse > 0:
        ax.text(rmse + 0.15, i, f'RMSE {rmse:.3f}   R² {r2:.2f}',
                va='center', fontsize=10, fontweight='bold')
    else:
        ax.text(0.15, i, 'RMSE 0.000   R² 1.00',
                va='center', fontsize=10, fontweight='bold', color=C_NEU)

# Highlight where we land
our_y = 2
ax.axhline(our_y - 0.5, color=C_GOOD, linestyle=':', alpha=0.4, linewidth=1)
ax.axhline(our_y + 0.5, color=C_GOOD, linestyle=':', alpha=0.4, linewidth=1)

plt.tight_layout()
plt.savefig(OUT / "impl_01_predictability_ladder.png")
plt.close()


# ---------------------------------------------------------------------------
# Implications chart 2: Distance traveled vs distance remaining
# ---------------------------------------------------------------------------
naive = 13.877
ours = 9.527
academic = 8.54
ceiling = 7.5

# Stages
distances = pd.DataFrame([
    {'stage': 'Distance traveled\n(naive → our model)',
     'value': naive - ours, 'pct': (naive - ours)/naive*100, 'color': C_GOOD},
    {'stage': 'Gap to academic SOTA\n(more features/tuning)',
     'value': ours - academic, 'pct': (ours - academic)/naive*100, 'color': C_HI2},
    {'stage': 'Gap to ceiling w/ real-time data\n(injuries, lineups, live Vegas)',
     'value': academic - ceiling, 'pct': (academic - ceiling)/naive*100, 'color': '#9467bd'},
    {'stage': 'Irreducible noise\n(can never be closed)',
     'value': ceiling, 'pct': ceiling/naive*100, 'color': C_NEU},
])

fig, ax = plt.subplots(figsize=(12, 4))
left = 0
for _, row in distances.iterrows():
    ax.barh(0, row['value'], left=left, color=row['color'],
            edgecolor='black', linewidth=0.5, alpha=0.85, label=row['stage'])
    ax.text(left + row['value']/2, 0, f'{row["pct"]:.1f}%',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='white' if row['color'] in [C_GOOD, C_NEU, '#9467bd'] else 'black')
    left += row['value']

ax.set_xlim(0, naive)
ax.set_ylim(-0.6, 0.6)
ax.set_yticks([])
ax.set_xlabel('Test RMSE (fantasy points) — total span: 0 (perfect) to 13.877 (naive)')
ax.set_title('Where the remaining error lives — and which parts are reachable')
ax.spines['left'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.45), ncol=2, fontsize=9)

# Key markers
for x, label in [(0, 'Perfect'), (ceiling, 'Theoretical\nceiling'),
                  (academic, 'Academic\nSOTA'), (ours, 'Our\nmodel'), (naive, 'Naive')]:
    ax.axvline(x, color='black', linewidth=0.5, alpha=0.3)
    ax.text(x, -0.3, label, ha='center', va='top', fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "impl_02_error_composition.png")
plt.close()


print(f"Generated charts in: {OUT}")
for p in sorted(OUT.glob("*.png")):
    print(f"  {p.name}")
