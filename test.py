import jax
import jaxmoseq
import yaml

jax.random.PRNGKey(0)

config_path = "hyperparams.yml"

with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
        
print(config)