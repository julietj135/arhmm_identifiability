import yaml
import h5py
from run import *
import tqdm as tqdm
from jaxmoseq.utils import get_durations, get_frequencies
from utils import load_checkpoint
import os

def plot_progress(
    model,
    data,
    checkpoint_path,
    iteration,
    project_dir=None,
    model_name=None,
    path=None,
    savefig=True,
    fig_size=None,
    window_size=600,
    min_frequency=0.001,
    min_histogram_length=10,
):
    z = np.array(model["states"]["z"])
    mask = np.array(data["mask"])
    durations = get_durations(z, mask)
    frequencies = get_frequencies(z, mask)

    with h5py.File(checkpoint_path, "r") as f:
        saved_iterations = np.sort([int(i) for i in f["model_snapshots"]])

    fig, axs = plt.subplots(4,1)
    fig_size = (10,10)
        
    frequencies = np.sort(frequencies[frequencies > min_frequency])[::-1]
    xmax = max(len(frequencies), min_histogram_length)
    axs[0].bar(range(len(frequencies)), frequencies, width=1)
    axs[0].set_ylabel("frequency\ndistribution")
    # axs[0].set_xlabel("syllable")
    axs[0].set_xlim([-1, xmax + 1])
    axs[0].set_yticks([])

    lim = int(np.percentile(durations, 95))
    lim = 80 # so x axis ticks are same for all kappas
    binsize = max(int(np.floor(lim / 30)), 1)
    axs[1].hist(durations, range=(1, lim), bins=(int(lim / binsize)), density=True)
    axs[1].set_xlim([1, lim])
    # axs[1].set_xlabel("syllable duration (frames)")
    axs[1].set_ylabel("duration\ndistribution")
    axs[1].set_yticks([])

    if len(saved_iterations) > 1:
        window_size = int(min(window_size, mask.max(0).sum() - 1))
        nz = np.stack(np.array(mask[:, window_size:]).nonzero(), axis=1)
        batch_ix, start = nz[np.random.randint(nz.shape[0])]

        sample_state_history = []
        median_durations = []

        for i in saved_iterations:
            with h5py.File(checkpoint_path, "r") as f:
                z = np.array(f[f"model_snapshots/{i}/states/z"])
                sample_state_history.append(z[batch_ix, start : start + window_size])
                median_durations.append(np.median(get_durations(z, mask)))

        axs[2].scatter(saved_iterations, median_durations)
        axs[2].set_ylim([-1, 50])
        # axs[2].set_xlabel("iteration")
        axs[2].set_ylabel("median\nduration")
        axs[2].set_yticks([])
        
        axs[3].imshow(
            sample_state_history,
            cmap=plt.cm.jet,
            aspect="auto",
            interpolation="nearest",
        )
        axs[3].set_xlabel("Time (frames)")
        axs[3].set_ylabel("Iterations")
        axs[3].set_title("State sequence history")

        yticks = [
            int(y) for y in axs[3].get_yticks() if y < len(saved_iterations) and y > 0
        ]
        yticklabels = saved_iterations[yticks]
        axs[3].set_yticks(yticks)
        axs[3].set_yticklabels(yticklabels)


    title = f"Iteration {iteration}"
    if model_name is not None:
        title = f"{model_name}: {title}"
    fig.set_size_inches(fig_size)
    plt.tight_layout()

    if not os.path.isdir(project_dir+"/"+model_name+"/figures/"):
        os.mkdir(project_dir+"/"+model_name+"/figures/")
    path = project_dir+"/"+model_name+"/figures/fitting_dist_{}.pdf".format(model_name)
    plt.savefig(path)
    return fig, axs


def get_sequences(save_dir,
                  prefix,
                  data,
                  subject_id = 0,
                  **kwargs
                  ):
    # get model names
    model_names=[]
    for subdir, dirs, files in os.walk(save_dir):
        if len(subdir.split("/")[-1])>0 and prefix in subdir.split("/")[-1]:
            model_names.append(subdir.split("/")[-1])
    model_names.sort()
    
    num_subjects, num_frames, num_keypoints, d = data["Y"].shape[0], data["Y"].shape[1], data["Y"].shape[2], data["Y"].shape[3]

    # plot sequences for all models in one graph
    fig, axs = plt.subplots(len(model_names), 1, figsize=(10,20))  # Adjust figsize as needed
    for i, model_name in enumerate(model_names):
        plot_states(save_dir, model_name, subject_id, 0, num_frames, axs[i])
    fig.tight_layout()
  
    print("saving in ", save_dir + "/figures/state_sequence_{}.pdf".format(key))
    fig.savefig(save_dir + "/figures/state_sequence_{}.pdf".format(key))
    
def plot_states(save_dir, 
                model_name, 
                name,
                start, 
                window_size, 
                ax):
    model_dir = os.path.join(save_dir, model_name)
    
    if not os.path.isdir(model_dir+"/figures/"):
        os.mkdir(model_dir+"/figures/")
    
    model, data, metadata, current_iter = load_checkpoint(save_dir, model_name)
    mask = np.array(data["mask"])
    window_size = int(min(window_size, mask[index].sum() - 1))
    print("Shape of mask: ", mask.shape, " same as shape of 'z'")
        
    sample_state_history = []
    with h5py.File(f"{model_dir}/checkpoint.h5", "r") as f:
        saved_iterations = np.sort([int(i) for i in f["model_snapshots"]])
        
        for i in saved_iterations:
            z = f[f"model_snapshots/{i}/states/z"][()]
            sample_state_history.append(z[index, start : start + window_size])
    print("The mask for this segment/subject is {} frames".format(mask[index].sum()))
    ax.imshow(sample_state_history, cmap = plt.cm.jet, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Iterations")
    ax.set_title("State sequence history for {}, {}".format(name, model_name))
    yticks = [int(y) for y in ax.get_yticks() if y <= len(saved_iterations) and y > 0]
    xticks = [int(x) for x in ax.get_xticks() if x <= start + window_size and x >= 0]
    yticklabels = saved_iterations[yticks]
    xticklabels = np.array(xticks)+np.ones(len(xticks))*start
    xticklabels = xticklabels.astype(int)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)