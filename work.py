import numpy as np
from data import *
import jaxmoseq as jxmq
from run import *
from utils import *
from viz import *

id = 1
body = 'l'
num_windows = 20

sub = get_raw_data(body, id)
subs = augment_data(sub, id, num_windows, body)

coordinates = {}

for i in range(num_windows):
        coordinates[i] = subs[i]

# unpack hyperparams
config_path = "jaxmoseq/hyperparams.yml"

with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
        
project_dir = "outputs/TWO-STATES"

for num in np.arange(5):
        print("SEED: ", num)
        
        seed = jax.random.PRNGKey(num)

        # initialize model, has PCA
        model, data = init_model(coordinates, **config, seed=seed, whiten=True)

        # run AR model
        model_name = "TRIAL-" + str(seed)
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
        full_interations = 100 
        model, model_name = fit_model(project_dir,
                model_name,
                model,
                seed, 
                data,
                1000.0,
                full_interations,
                ar_only=False,
                save_every_n_iters=20)

# plotting
checkpoint_path = "{}/{}/checkpoint.h5".format(project_dir,model_name)
total_iterations = full_interations+AR_interations
plot_progress(model,
                data,
                checkpoint_path,
                total_iterations,
                project_dir,
                model_name=model_name)
get_sequences(project_dir,
                  "TRIAL",
                  data)