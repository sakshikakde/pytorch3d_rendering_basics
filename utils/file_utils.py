import matplotlib.pyplot as plt

def save_fig(tensor, filename):
    for i, t in enumerate(tensor):
        array = tensor[i].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(array)
        plt.savefig(f"{filename}_{i}.png")