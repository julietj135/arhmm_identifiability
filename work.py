import numpy as np
from data import *
import jaxmoseq as jxmq
from run import *
from utils import *
from viz import *
import jax
import jax.numpy as jnp
from jaxmoseq.models.arhmm.gibbs import *
from jaxmoseq.models.slds.log_prob import log_joint_likelihood
from jaxmoseq.models.arhmm.log_prob import marginal_log_likelihood
from tqdm import tqdm

print(jax.devices())

id = 1
body = 'l'
num_windows = 20

# unpack hyperparams
config_path = "jaxmoseq/hyperparams.yml"

with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
print(config)
project_dir = "outputs/NOISE-TESTS/HIGHER-K0"

########### GET DATA ############
sub = get_raw_data(body, id)
subs = augment_data(sub, id, num_windows, body)

coordinates = {}

for i in range(num_windows):
        coordinates[i] = subs[i]
        
########### FITTING MODELS ##########
for num in [0,1,12,10,18,11,13,15,16,4]:
        print("SEED: ", num)
        
        seed = jax.random.PRNGKey(num)

        # initialize model, has PCA
        model, data = init_model(coordinates, **config, seed=seed, whiten=True)

        # run AR model
        model_name = "TRIAL-" + str(num)
        model = update_hypparams(model, kappa = 100.0)

        AR_interations = 50
        model, model_name = fit_model(project_dir,
                model_name,
                model,
                seed, 
                data,
                100.0,
                AR_interations,
                ar_only=True,
                save_every_n_iters=20)

        # run whole model
        model = update_hypparams(model, kappa = 1000.0)
        full_interations = 400
        model, model_name = fit_model(project_dir,
                model_name,
                model,
                seed, 
                data,
                1000.0,
                full_interations,
                ar_only=False,
                save_every_n_iters=20)

######### PLOTTING FITTED MODELS ########
# model, data = init_model(coordinates, **config, seed=0, whiten=True)
# model_names = [0,1,12,17,5,6,7,8,11,13,15,16,4,10,14,18,19,2,3,9]
model_names = [0,1,12,10,18,11,13,15,16,4]
for i in range(len(model_names)):
        model_names[i] = "TRIAL-" + str(model_names[i])
get_sequences(project_dir,
                  "TRIAL",
                  data,
                  model_names = model_names)

######### LOAD MODEL FOR POSTERIORS #######
# for i in [12,17,5,6,7]:
#     model_name = f"TRIAL-{i}"
#     save_name = f"TWO-STATES/{model_name}/figures"
#     model, data, iteration = load_checkpoint(project_dir, model_name)
#     n_samp = 10000
#     num_states = model["hypparams"]["ar_hypparams"]['num_states']
#     latent_dim = model["hypparams"]["ar_hypparams"]['latent_dim']
#     ar_dim = model["hypparams"]["ar_hypparams"]['K_0'].shape[0]
#     pi_samples = np.zeros((n_samp,num_states,num_states))
#     beta_samples = np.zeros((n_samp,num_states))
#     Ab_samples = np.zeros((n_samp,num_states,latent_dim,ar_dim))
#     Q_samples = np.zeros((n_samp,num_states,latent_dim,latent_dim))
#     for i in tqdm(range(n_samp)):
#             seed = jax.random.PRNGKey(i)
#             beta_samples[i], pi_samples[i] = resample_hdp_transitions(
#                 seed, **data, **model["states"], **model["params"], **model["hypparams"]["trans_hypparams"])
            
#             Ab_samples[i], Q_samples[i] = resample_ar_params(
#                 seed, **data, **model["states"], **model["params"], **model["hypparams"]["ar_hypparams"]
#             )


#     fig, axs = plt.subplots(2, 1, figsize=(6, 8))

#     axs[0].hist(beta_samples[:, 0], bins=15, density=True)
#     axs[0].set_title("State 0")
#     axs[0].set_xlabel("Value")
#     axs[0].set_ylabel("Density")

#     axs[1].hist(beta_samples[:, 1], bins=15, density=True)
#     axs[1].set_title("State 1")
#     axs[1].set_xlabel("Value")
#     axs[1].set_ylabel("Density")

#     # Improve layout and save
#     plt.tight_layout()
#     plt.savefig(f"outputs/{save_name}/beta_samples.pdf")

#     fig, axs = plt.subplots(num_states,num_states, figsize=(10, 8))
#     entries = [(i, j) for i in range(num_states) for j in range(num_states)]
#     for idx, (i, j) in enumerate(entries):
#         ax = axs[idx // num_states, idx % num_states]  
#         ax.hist(pi_samples[:, i, j], bins=15, density=True)
#         ax.set_title(f"Entry ({i},{j})")
#         ax.set_xlabel("Value")
#         ax.set_ylabel("Density")

#     plt.tight_layout() 
#     plt.savefig(f"outputs/{save_name}/pi_samples_all.pdf") 

#     plt.figure()
#     fig, axes = plt.subplots(latent_dim,ar_dim, figsize=(30, 10))  
#     entries_to_plot = [(i, j) for i in range(latent_dim) for j in range(ar_dim)]  
#     axes = axes.flatten()
#     for idx, (i, j) in enumerate(entries_to_plot):
#             ax = axes[idx] 
#             ax.hist(Ab_samples[:, 0, i, j], bins=15, density=True)
#             ax.set_title(f"Entry ({i},{j})")
#             ax.set_xlabel("Value")
#             ax.set_ylabel("Density")

#     plt.title("State 0")
#     plt.tight_layout()
#     plt.savefig(f"outputs/{save_name}/Ab_samples_state_0.pdf")

#     plt.figure()
#     fig, axes = plt.subplots(latent_dim,ar_dim, figsize=(30, 10))  # Adjust size for clarity
#     axes = axes.flatten()
#     for idx, (i, j) in enumerate(entries_to_plot):
#             ax = axes[idx] 
#             ax.hist(Ab_samples[:, 1, i, j], bins=15, density=True)
#             ax.set_title(f"Entry ({i},{j})")
#             ax.set_xlabel("Value")
#             ax.set_ylabel("Density")

#     plt.title("State 1")
#     plt.tight_layout()
#     plt.savefig(f"outputs/{save_name}/Ab_samples_state_1.pdf")

#     plt.figure()
#     fig, axes = plt.subplots(latent_dim,latent_dim, figsize=(12, 12))  # Adjust size for clarity
#     entries_to_plot = [(i, j) for i in range(latent_dim) for j in range(latent_dim)]  # Example: 3x3 grid

#     for idx, (i, j) in enumerate(entries_to_plot):
#         ax = axes[idx // latent_dim, idx % latent_dim]
#         ax.hist(Q_samples[:, 0, i, j], bins=15)
#         ax.set_title(f"Entry ({i},{j})")

#     plt.title("State 0")
#     plt.tight_layout()
#     plt.savefig(f"outputs/{save_name}/Q_samples_state_0.pdf")

#     plt.figure()
#     fig, axes = plt.subplots(latent_dim,latent_dim, figsize=(12, 12))  # Adjust size for clarity
#     entries_to_plot = [(i, j) for i in range(latent_dim) for j in range(latent_dim)]  # Example: 3x3 grid

#     for idx, (i, j) in enumerate(entries_to_plot):
#         ax = axes[idx // latent_dim, idx % latent_dim]
#         ax.hist(Q_samples[:, 1, i, j], bins=15)
#         ax.set_title(f"Entry ({i},{j})")

#     plt.title("State 1")
#     plt.tight_layout()
#     plt.savefig(f"outputs/{save_name}/Q_samples_state_1.pdf")

    
# for model_name in model_names:
#     model, data, iteration = load_checkpoint(project_dir, model_name)
#     print(model_name, model["params"]["Ab"])

# for model_name in model_names:
#     model, data, iteration = load_checkpoint(project_dir, model_name)
#     print(model_name, model["params"]["Q"])

# ########### COMPARE POSTERIOR DISTRIBUTIONS ACROSS MODEL SEEDS ########
save_name = "NOISE-TESTS/HIGHER-K0/figures"
fig1, axes1 = plt.subplots(2,2, figsize=(6,6))  
fig2, axes2 = plt.subplots(2,5, figsize=(15,6))  
axes2 = axes2.flatten()
fig3, axes3 = plt.subplots(2,5, figsize=(16,5))  
axes3 = axes3.flatten()
fig4, axes4 = plt.subplots(2,5, figsize=(16,5))  
axes4 = axes4.flatten()
fig5, axes5 = plt.subplots(2,5, figsize=(15,6))  
axes5 = axes5.flatten()
fig6, axes6 = plt.subplots(2,5, figsize=(15,6))  
axes6 = axes6.flatten()
colors = ["r"]*5+["b"]*5
for idx,i in enumerate([0,1,12,10,18,11,13,15,16,4]):
    model_name = f"TRIAL-{i}"
    model, data, iteration = load_checkpoint(project_dir, model_name)
    n_samp = 5000
    num_states = model["hypparams"]["ar_hypparams"]['num_states']
    latent_dim = model["hypparams"]["ar_hypparams"]['latent_dim']
    ar_dim = model["hypparams"]["ar_hypparams"]['K_0'].shape[0]
    pi_samples = np.zeros((n_samp,num_states,num_states))
    beta_samples = np.zeros((n_samp,num_states))
    Ab_samples = np.zeros((n_samp,num_states,latent_dim,ar_dim))
    Q_samples = np.zeros((n_samp,num_states,latent_dim,latent_dim))
    for s in tqdm(range(n_samp)):
            seed = jax.random.PRNGKey(s)
            beta_samples[s], pi_samples[s] = resample_hdp_transitions(
                seed, **data, **model["states"], **model["params"], **model["hypparams"]["trans_hypparams"])
        
            Ab_samples[s], Q_samples[s] = resample_ar_params(
                seed, **data, **model["states"], **model["params"], **model["hypparams"]["ar_hypparams"]
            )
    
    for j in range(2):
        for k in range(2):
            axes1[j,k].hist(pi_samples[:,j,k], alpha = 0.5, density=True, color=colors[idx])
    
    img = axes2[idx].imshow(model["params"]["pi"])
    img.set_clim(vmin=0, vmax=1)
    # fig2.colorbar(img, ax=axes2[idx])
    
    img = axes3[idx].imshow(model["params"]["Ab"][0])
    # fig3.colorbar(img, ax=axes3[idx])
    img = axes4[idx].imshow(model["params"]["Ab"][1])
    # fig4.colorbar(img, ax=axes4[idx])
    
    img = axes5[idx].imshow(model["params"]["Q"][0])
    img.set_clim(vmin=-0.003, vmax=0.007)
    # fig5.colorbar(img, ax=axes5[idx])
    img = axes6[idx].imshow(model["params"]["Q"][1])
    img.set_clim(vmin=-0.003, vmax=0.008)
    # fig6.colorbar(img, ax=axes6[idx])

fig1.tight_layout()
fig1.savefig(f"outputs/{save_name}/pi_samples_hist_colorlabeled.pdf")

fig2.tight_layout()
fig2.savefig(f"outputs/{save_name}/pi_matrix_wo_cb.pdf")

fig3.tight_layout()
fig3.savefig(f"outputs/{save_name}/Ab_matrix_state0_wo_cb.pdf")

fig4.tight_layout()
fig4.savefig(f"outputs/{save_name}/Ab_matrix_state1_wo_cb.pdf")

fig5.tight_layout()
fig5.savefig(f"outputs/{save_name}/Q_matrix_state0_wo_cb.pdf")

fig6.tight_layout()
fig6.savefig(f"outputs/{save_name}/Q_matrix_state1_wo_cb.pdf")


########### FIND INDIVIDUAL TRACES ACROSS ITERATIONS FOR EACH TRIAL ########
# for idx,i in enumerate([0,1,12,10,18,11,13,15,16,4]):
#     num_iterations = 21
#     model_name = f"TRIAL-{i}"
#     model, data, iteration = load_checkpoint(project_dir, model_name)
#     save_name = f"TWO-STATES/{model_name}/figures"
    
#     # hyperparams
#     num_states = model["hypparams"]["ar_hypparams"]['num_states']
#     latent_dim = model["hypparams"]["ar_hypparams"]['latent_dim']
#     ar_dim = model["hypparams"]["ar_hypparams"]['K_0'].shape[0]
    
#     # params
#     pi_samples = np.zeros((num_iterations,num_states,num_states))
#     beta_samples = np.zeros((num_iterations,num_states))
#     Ab_samples = np.zeros((num_iterations,num_states,latent_dim,ar_dim))
#     Q_samples = np.zeros((num_iterations,num_states,latent_dim,latent_dim))
    
#     # probabilities
#     log_joint_likelihoods = np.zeros((num_iterations, 4)) # total log probability for each latent state
#     marginal_log_likelihoods = np.zeros((num_iterations)) # marginal log likelihood of continuous latents given model parameters. log(P(state seq | params)
        
#     for index, iteration in enumerate(np.arange(0,420,20)):
#         model, data, iteration = load_checkpoint(project_dir, model_name, iteration=iteration)
#         model["hypparams"]["obs_hypparams"]["s_0"] =  model["hypparams"]["obs_hypparams"]["sigmasq_0"]
#         data["Y"] = data["Y"].reshape(data["Y"].shape[0], data["Y"].shape[1], -1)

#         ll = log_joint_likelihood(
#             **data, 
#             **model["states"], 
#             **model["params"], 
#             **model["hypparams"]["obs_hypparams"]
#         )
#         log_joint_likelihoods[index] = [ll["Y"], ll["s"], ll["x"], ll["z"]]
        
#         marginal_log_likelihoods[index] = marginal_log_likelihood(
#             data["mask"], 
#             **model["states"], 
#             **model["params"]
#         )
#         pi_samples[index], beta_samples[index] = model["params"]["pi"], model["params"]["betas"]
#         Ab_samples[index], Q_samples[index] = model["params"]["Ab"], model["params"]["Q"]
        
#     plt.figure(figsize=(4,4))
#     plt.plot(np.arange(0,420,20), beta_samples[:, 0], label="State 0", color="blue")
#     plt.plot(np.arange(0,420,20), beta_samples[:, 1], label="State 1", color="black")
#     plt.title("Beta Samples")
#     plt.xlabel("Iteration")
#     plt.ylabel("Trace")
#     plt.tight_layout()
#     plt.savefig(f"outputs/{save_name}/beta_samples_trace.pdf")

#     fig, axs = plt.subplots(num_states,num_states, figsize=(10, 8))
#     entries = [(i, j) for i in range(num_states) for j in range(num_states)]
#     for idx, (i, j) in enumerate(entries):
#         ax = axs[idx // num_states, idx % num_states]  
#         ax.plot(np.arange(0,420,20), pi_samples[:, i, j], color="blue")
#         ax.set_title(f"Entry ({i},{j})")
#         ax.set_xlabel("Iteration")
#         ax.set_ylabel("Trace")

#     plt.tight_layout() 
#     plt.savefig(f"outputs/{save_name}/pi_samples_all_trace.pdf") 

#     plt.figure()
#     fig, axes = plt.subplots(latent_dim,ar_dim, figsize=(30, 10))  
#     entries_to_plot = [(i, j) for i in range(latent_dim) for j in range(ar_dim)]  
#     axes = axes.flatten()
#     for idx, (i, j) in enumerate(entries_to_plot):
#             ax = axes[idx] 
#             ax.plot(np.arange(0,420,20), Ab_samples[:, 0, i, j], label="state 0", color="blue")
#             ax.plot(np.arange(0,420,20), Ab_samples[:, 1, i, j], label="state 1", color="black")
#             ax.set_title(f"Entry ({i},{j})")
#             ax.set_xlabel("Iteration")
#             ax.set_ylabel("Trace")

#     plt.tight_layout()
#     plt.savefig(f"outputs/{save_name}/Ab_samples_all_trace.pdf")

#     plt.figure()
#     fig, axes = plt.subplots(latent_dim,latent_dim, figsize=(12, 12))  # Adjust size for clarity
#     entries_to_plot = [(i, j) for i in range(latent_dim) for j in range(latent_dim)]  # Example: 3x3 grid

#     for idx, (i, j) in enumerate(entries_to_plot):
#         ax = axes[idx // latent_dim, idx % latent_dim]
#         ax.plot(np.arange(0,420,20), Q_samples[:, 0, i, j], label="state 0", color="blue")
#         ax.plot(np.arange(0,420,20), Q_samples[:, 1, i, j], label="state 1", color="black")
#         ax.set_title(f"Entry ({i},{j})")

#     plt.tight_layout()
#     plt.savefig(f"outputs/{save_name}/Q_samples_all_trace.pdf")

#     fig, axs = plt.subplots(2,2, figsize=(6, 6))
#     axs = axs.flatten()
#     for q, label in enumerate(["Y", "s", "x", "z"]):
#         axs[q].plot(np.arange(0,420,20), log_joint_likelihoods[:,q], label=label, color="blue")
#         axs[q].set_xlabel("Iteration")
#         axs[q].set_ylabel("Trace")
#         axs[q].set_title(f"Log Joint Likelihoods for {label}")
#     plt.tight_layout()
#     plt.savefig(f"outputs/{save_name}/log_joint_likelihoods_trace.pdf")
    
#     plt.figure()
#     plt.plot(np.arange(0,420,20), marginal_log_likelihoods, label="Marginal Log Likelihoods", color="blue")
#     plt.xlabel("Iteration")
#     plt.ylabel("Trace")
#     plt.title(f"Marginal Log Likelihoods")
#     plt.tight_layout()
#     plt.savefig(f"outputs/{save_name}/marginal_log_likelihoods_trace.pdf")

# ########### FIND INDIVIDUAL TRACES ACROSS TRIALS ########
save_name = "NOISE-TESTS/HIGHER-K0/figures/traces"
fig1, axes1 = plt.subplots(2,2, figsize=(6,6))  # pis
fig2, axes2 = plt.subplots(2,1, figsize=(6,4))  # betas
fig3, axes3 = plt.subplots(latent_dim,ar_dim, figsize=(20, 8)) # Ab state 0
fig4, axes4 = plt.subplots(latent_dim,ar_dim, figsize=(20, 8)) # Ab state 1
fig5, axes5 = plt.subplots(latent_dim,latent_dim, figsize=(10, 10)) # Q state 0
fig6, axes6 = plt.subplots(latent_dim,latent_dim, figsize=(10, 10)) # Q state 1
fig7, axes7 = plt.subplots(1,1, figsize=(4,4)) # marginals
fig8, axes8 = plt.subplots(2,2, figsize=(6,6))  # log joint likelihoods
colors = ["r"]*5+["b"]*5
num_iterations = 21

for idx,i in enumerate([0,1,12,10,18,11,13,15,16,4]):
    
    model_name = f"TRIAL-{i}"
    model, data, iteration = load_checkpoint(project_dir, model_name)
    
    # hyperparams
    num_states = model["hypparams"]["ar_hypparams"]['num_states']
    latent_dim = model["hypparams"]["ar_hypparams"]['latent_dim']
    ar_dim = model["hypparams"]["ar_hypparams"]['K_0'].shape[0]
    
    # params
    pi_samples = np.zeros((num_iterations,num_states,num_states))
    beta_samples = np.zeros((num_iterations,num_states))
    Ab_samples = np.zeros((num_iterations,num_states,latent_dim,ar_dim))
    Q_samples = np.zeros((num_iterations,num_states,latent_dim,latent_dim))
    
    # probabilities
    log_joint_likelihoods = np.zeros((num_iterations, 4)) # total log probability for each latent state
    marginal_log_likelihoods = np.zeros((num_iterations)) # marginal log likelihood of continuous latents given model parameters. log(P(state seq | params)
        
    for index, iteration in enumerate(np.arange(0,420,20)):
        model, data, iteration = load_checkpoint(project_dir, model_name, iteration=iteration)
        model["hypparams"]["obs_hypparams"]["s_0"] =  model["hypparams"]["obs_hypparams"]["sigmasq_0"]
        data["Y"] = data["Y"].reshape(data["Y"].shape[0], data["Y"].shape[1], -1)

        ll = log_joint_likelihood(
            **data, 
            **model["states"], 
            **model["params"], 
            **model["hypparams"]["obs_hypparams"]
        )
        log_joint_likelihoods[index] = [ll["Y"], ll["s"], ll["x"], ll["z"]]
        
        marginal_log_likelihoods[index] = marginal_log_likelihood(
            data["mask"], 
            **model["states"], 
            **model["params"]
        )
        pi_samples[index], beta_samples[index] = model["params"]["pi"], model["params"]["betas"]
        Ab_samples[index], Q_samples[index] = model["params"]["Ab"], model["params"]["Q"]
        

    # plot pi samples
    entries = [(k, l) for k in range(num_states) for l in range(num_states)]
    for kdx, (k, l) in enumerate(entries):
        ax = axes1[kdx // num_states, kdx % num_states]  
        ax.plot(np.arange(0,420,20), pi_samples[:, k, l], color=colors[idx])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Trace")
    
    # plot beta samples
    axes2[0].plot(np.arange(0,420,20), beta_samples[:, 0], label="State 0", color=colors[idx])
    axes2[0].set_ylabel("Trace")
    axes2[0].set_title(f"Beta Samples State 0")
    axes2[1].plot(np.arange(0,420,20), beta_samples[:, 1], label="State 1", color=colors[idx])
    axes2[1].set_ylabel("Trace")
    axes2[1].set_title(f"Beta Samples State 1")
    
    # plot Ab samples
    entries_to_plot = [(k, l) for k in range(latent_dim) for l in range(ar_dim)]
    axes3 = axes3.flatten()
    axes4 = axes4.flatten()
    for kdx, (k, l) in enumerate(entries_to_plot):
            ax = axes3[kdx] 
            ax.plot(np.arange(0,420,20), Ab_samples[:, 0, k, l], label="state 0", color=colors[idx])
            
            ax = axes4[kdx]
            ax.plot(np.arange(0,420,20), Ab_samples[:, 1, k, l], label="state 1", color=colors[idx])
    
    # plot Q samples
    entries_to_plot = [(k, l) for k in range(latent_dim) for l in range(latent_dim)]
    axes5 = axes5.flatten()
    axes6 = axes6.flatten()
    for kdx, (k, l) in enumerate(entries_to_plot):
        ax = axes5[kdx] 
        ax.plot(np.arange(0,420,20), Q_samples[:, 0, k, l], label="state 0", color=colors[idx])
        
        ax = axes6[kdx]
        ax.plot(np.arange(0,420,20), Q_samples[:, 1, k, l], label="state 1", color=colors[idx])
    
    # plot marginals
    axes7.plot(np.arange(0,420,20), marginal_log_likelihoods, label="Marginal Log Likelihoods", color=colors[idx])
    axes7.set_xlabel("Iteration")
    axes7.set_ylabel("Trace")
    axes7.set_title(f"Marginal Log Likelihoods")
    
    # plot log joint likelihoods
    entries = [(k, l) for k in range(num_states) for l in range(num_states)]
    for q, (k, l), label in zip(np.arange(4), entries, ["Y", "s", "x", "z"]):
        ax = axes8[q // num_states, q % num_states] 
        ax.plot(np.arange(0,420,20), log_joint_likelihoods[:,q], label=label, color=colors[idx])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Trace")
        ax.set_title(f"Log Joint Likelihoods for {label}")
        
fig1.tight_layout()
fig1.savefig(f"outputs/{save_name}/pi_traces.pdf")

fig2.tight_layout()
fig2.savefig(f"outputs/{save_name}/betas_traces.pdf")

fig3.tight_layout()
fig3.savefig(f"outputs/{save_name}/Ab_state_0_traces.pdf")

fig4.tight_layout()
fig4.savefig(f"outputs/{save_name}/Ab_state_1_traces.pdf")

fig5.tight_layout()
fig5.savefig(f"outputs/{save_name}/Q_state_0_traces.pdf")

fig6.tight_layout()
fig6.savefig(f"outputs/{save_name}/Q_state_1_traces.pdf")

fig7.tight_layout()
fig7.savefig(f"outputs/{save_name}/marginals_traces.pdf")

fig8.tight_layout()
fig8.savefig(f"outputs/{save_name}/log_joint_likelihoods_traces.pdf")
