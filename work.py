import numpy as np
from data import *
import jaxmoseq as jxmq
from run import *

id = 1
body = 'l'
seed = jax.random.PRNGKey(0)
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

# initialize model, has PCA
model = init_model(coordinates, **config, seed=seed, whiten=True)

print(model)
# run AR model

# run whole model