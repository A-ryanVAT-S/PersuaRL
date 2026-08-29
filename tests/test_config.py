"""Config loading: defaults chains, dotted access, env expansion, --set overrides."""

from __future__ import annotations

import pytest

from persuarl.config import Config, ConfigError, load_config


@pytest.fixture
def configs(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "model:\n  id: base-model\n  dtype: auto\ntrain:\n  epochs: 1\n", encoding="utf-8"
    )
    (tmp_path / "child.yaml").write_text(
        "defaults: [base.yaml]\nmodel:\n  dtype: bfloat16\ntrain:\n  epochs: 3\n", encoding="utf-8"
    )
    return tmp_path


def test_defaults_chain_merges_deeply(configs):
    config = load_config(configs / "child.yaml")
    assert config.get("model.id") == "base-model"      # inherited
    assert config.get("model.dtype") == "bfloat16"     # overridden
    assert config.get("train.epochs") == 3


def test_missing_key_raises_unless_a_default_is_given(configs):
    config = load_config(configs / "base.yaml")
    with pytest.raises(ConfigError, match="missing required config key"):
        config.get("model.nonexistent")
    assert config.get("model.nonexistent", "fallback") == "fallback"


def test_explicit_none_default_is_distinguishable_from_missing(configs):
    """`default=None` must not be mistaken for "no default supplied"."""
    config = load_config(configs / "base.yaml")
    assert config.get("model.nonexistent", None) is None


def test_overrides_preserve_python_types(configs):
    config = load_config(
        configs / "child.yaml",
        ["train.epochs=5", "train.lr=2e-5", "train.flag=True", "model.id=custom/model"],
    )
    assert config.get("train.epochs") == 5
    assert config.get("train.lr") == pytest.approx(2e-5)
    assert config.get("train.flag") is True
    assert config.get("model.id") == "custom/model"    # unquoted string stays a string


def test_override_can_create_a_new_section(configs):
    config = load_config(configs / "base.yaml", ["brand.new.key=1"])
    assert config.get("brand.new.key") == 1


def test_malformed_override_is_rejected(configs):
    with pytest.raises(ConfigError, match="key=value"):
        load_config(configs / "base.yaml", ["not-an-assignment"])


def test_env_expansion_uses_the_fallback_when_unset(configs, monkeypatch):
    monkeypatch.delenv("PERSUARL_TEST_ROOT", raising=False)
    (configs / "env.yaml").write_text(
        'path: "${PERSUARL_TEST_ROOT:-/default/root}/data"\n', encoding="utf-8"
    )
    assert load_config(configs / "env.yaml").get("path") == "/default/root/data"


def test_env_expansion_prefers_the_environment(configs, monkeypatch):
    monkeypatch.setenv("PERSUARL_TEST_ROOT", "/scratch")
    (configs / "env.yaml").write_text(
        'path: "${PERSUARL_TEST_ROOT:-/default/root}/data"\n', encoding="utf-8"
    )
    assert load_config(configs / "env.yaml").get("path") == "/scratch/data"


def test_unset_variable_without_fallback_is_an_error(configs, monkeypatch):
    monkeypatch.delenv("PERSUARL_MISSING", raising=False)
    (configs / "env.yaml").write_text('path: "${PERSUARL_MISSING}"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="not set"):
        load_config(configs / "env.yaml")


def test_section_returns_an_empty_config_for_absent_paths():
    config = Config({"a": {"b": 1}})
    assert config.section("a").get("b") == 1
    assert config.section("nope").as_dict() == {}


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "absent.yaml")
