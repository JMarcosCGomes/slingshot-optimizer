import yaml

REQUIRED_ROLES = {"fixed", "planet", "probe"}

def load_config(path="params.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    active_celestial_bodies = [cb for cb in config["celestial_bodies"] if cb.get("active", True)]
    validate_config(active_celestial_bodies)
    config["celestial_bodies"] = active_celestial_bodies
    return config

def validate_config(bodies):
    active_roles = {b["role"] for b in bodies}
    missing = REQUIRED_ROLES - active_roles
    if missing:
        raise ValueError(f"Missing roles: {missing}")