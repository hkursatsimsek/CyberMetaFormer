"""Shared drawing helpers for the hand-drawn diagrams (`paper_figures.py` and the course report figures)."""
import os

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cybermetaformer.hidden_state.figures import BASE, GRID, INK, INK2, SERIES, SURFACE

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results', 'hidden_state')

RED, PALE, PALE2 = SERIES[1], '#eef4fc', '#f6f5f2'


def _box(ax, x, y, w, h, text, fc=SURFACE, ec=BASE, lw=1.2, fs=9, weight='normal', color=INK, ls='-', pad=0.02,
         ha='center'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f'round,pad={pad},rounding_size=0.12',
                                fc=fc, ec=ec, lw=lw, ls=ls))
    tx = x if ha == 'center' else x - w / 2 + 0.15
    ax.text(tx, y, text, ha=ha, va='center', fontsize=fs, fontweight=weight, color=color, linespacing=1.35)


def _arrow(ax, p, q, ls='-', color=INK2, lw=1.3, rad=0.0, style='-|>', ms=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))


def _region(ax, x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=0.25',
                                fc=PALE2, ec=GRID, lw=1.2))
    ax.text((x0 + x1) / 2, y1 - 0.32, title, ha='center', va='center', fontsize=10, fontweight='bold', color=INK2)
