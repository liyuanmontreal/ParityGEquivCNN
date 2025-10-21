import matplotlib.pyplot as plt
import numpy as np
import os

def save_triptych(Amean, Bmean, out_path, titleA="orig", titleB="mirror"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axs = plt.subplots(1,3, figsize=(9,3))
    axs[0].imshow(Amean, cmap='viridis'); axs[0].set_title(titleA); axs[0].axis('off')
    axs[1].imshow(Bmean, cmap='viridis'); axs[1].set_title(titleB); axs[1].axis('off')
    diff = np.abs(Amean - Bmean)
    axs[2].imshow(diff, cmap='hot'); axs[2].set_title("abs diff"); axs[2].axis('off')
    plt.tight_layout()
    plt.savefig(out_path); plt.close(fig)

def save_bar(layers, values, out_path, ylabel="correlation", title=None, ylim=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(6,3))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(layers)), layers, rotation=45, ha='right')
    if ylim: plt.ylim(*ylim)
    if ylabel: plt.ylabel(ylabel)
    if title: plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path); plt.close(fig)
