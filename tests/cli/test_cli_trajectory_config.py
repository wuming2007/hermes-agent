import copy


class _DummyAgent:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self._print_fn = None


def _make_config(cli_module, *, root=None, agent=None):
    cfg = copy.deepcopy(cli_module.CLI_CONFIG)
    if root is None:
        cfg.pop("save_trajectories", None)
    else:
        cfg["save_trajectories"] = root
    cfg.setdefault("agent", {})
    if agent is None:
        cfg["agent"].pop("save_trajectories", None)
    else:
        cfg["agent"]["save_trajectories"] = agent
    return cfg


def _init_shell_with_config(monkeypatch, cfg):
    import cli

    _DummyAgent.last_kwargs = None
    monkeypatch.setattr(cli, "CLI_CONFIG", cfg)
    monkeypatch.setattr(cli, "AIAgent", _DummyAgent)
    shell = cli.HermesCLI(model="test/model", compact=True, max_turns=1)
    monkeypatch.setattr(shell, "_ensure_runtime_credentials", lambda: True)
    shell._session_db = object()
    assert shell._init_agent() is True
    return shell, _DummyAgent.last_kwargs


def test_cli_passes_root_save_trajectories_config_to_agent(monkeypatch):
    import cli

    cfg = _make_config(cli, root=True, agent=None)
    shell, kwargs = _init_shell_with_config(monkeypatch, cfg)

    assert shell.save_trajectories is True
    assert kwargs["save_trajectories"] is True


def test_cli_agent_save_trajectories_overrides_root_config(monkeypatch):
    import cli

    cfg = _make_config(cli, root=True, agent=False)
    shell, kwargs = _init_shell_with_config(monkeypatch, cfg)

    assert shell.save_trajectories is False
    assert kwargs["save_trajectories"] is False


def test_cli_save_trajectories_defaults_false_when_unconfigured(monkeypatch):
    import cli

    cfg = _make_config(cli, root=None, agent=None)
    shell, kwargs = _init_shell_with_config(monkeypatch, cfg)

    assert shell.save_trajectories is False
    assert kwargs["save_trajectories"] is False
