# -*- coding: utf-8 -*-
# Emblema do robô MIYAGI — vetorial minimalista (sol nascente + robô com faixa + tendência)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Polygon
import numpy as np

RED, INK, STEEL, LIGHT = "#C63D2F", "#16181B", "#5B6770", "#EEF0F2"

def draw(dark_bg=False, fname="miyagi_emblem.png"):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_alpha(0.0)

    # Sol nascente (disco vermelho ao fundo)
    ax.add_patch(Circle((50, 58), 34, color=RED, zorder=1, alpha=0.95))

    # Cabeça do robô
    head_face = "#FFFFFF" if not dark_bg else LIGHT
    ax.add_patch(FancyBboxPatch((30, 38), 40, 34,
        boxstyle="round,pad=2,rounding_size=7",
        facecolor=head_face, edgecolor=INK, linewidth=3.2, zorder=3))
    # Antena
    ax.plot([50, 50], [74, 82], color=INK, lw=3.2, zorder=2)
    ax.add_patch(Circle((50, 84.5), 3.2, facecolor=RED, edgecolor=INK, lw=2.4, zorder=3))
    # Olhos (serenos: meia-lua para baixo, mestre zen)
    for cx in (41, 59):
        th = np.linspace(np.pi*0.15, np.pi*0.85, 40)
        ax.plot(cx + 6*np.cos(th), 50 + 4.5*np.sin(th), color=INK, lw=3.4,
                solid_capstyle="round", zorder=4)
    # Faixa de mestre (hachimaki) na testa
    ax.add_patch(Rectangle((28.2, 60), 43.6, 7.4, facecolor="#FFFFFF",
                 edgecolor=INK, lw=2.6, zorder=4))
    ax.add_patch(Circle((50, 63.7), 2.6, color=RED, zorder=5))
    # Pontas da faixa ao vento (à direita)
    ax.add_patch(Polygon([[71.5, 66.5], [86, 70], [79, 63.5]], closed=True,
                 facecolor="#FFFFFF", edgecolor=INK, lw=2.4, zorder=3))
    ax.add_patch(Polygon([[71.5, 62.5], [84, 58], [77, 56.5]], closed=True,
                 facecolor="#FFFFFF", edgecolor=INK, lw=2.4, zorder=3))

    # Linha de tendência (pincelada ascendente com degraus)
    x = [12, 26, 34, 46, 56, 70, 84]
    y = [16, 22, 19, 27, 24, 33, 40]
    ax.plot(x, y, color=INK, lw=5.2, solid_capstyle="round",
            solid_joinstyle="round", zorder=6)
    # Ponta de seta
    ax.add_patch(Polygon([[84, 40], [76.5, 39.2], [80.5, 33.2]], closed=True,
                 color=INK, zorder=6))

    fig.savefig(fname, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

draw(False, "miyagi_emblem.png")
draw(True,  "miyagi_emblem_dark.png")
print("emblemas gerados")
