import pytest
from hydra import compose, initialize_config_dir

from mamora.envs.factory import ENV_NAMES
from mamora.paths import CONF_DIR

ENV_OPTIONS = sorted(p.stem for p in (CONF_DIR / "env").glob("*.yaml"))


@pytest.mark.parametrize("env_option", ENV_OPTIONS)
def test_compose_every_env_option(env_option):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=[f"env={env_option}"])
    # `checkpoint_dir` is left unresolved on purpose: `${hydra:...}` only exists in a real run.
    assert cfg.env.name in ENV_NAMES
    assert cfg.rollout.steps > 0
    assert cfg.rollout.num_envs > 0
    assert cfg.dry_run is False
