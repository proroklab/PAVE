"""Smoke tests for eval package split and shim removal."""

import pytest


class TestEvalPackage:
    def test_eval_compare_loads(self):
        # ``critic-sharing eval`` is registered directly on the top-level
        # app as a single command (the old ``critic-sharing eval compare``
        # sub-namespace has been removed).
        from critic_sharing.cli.eval import eval_compare
        assert callable(eval_compare)

    def test_eval_command_registered_on_main_app(self):
        from critic_sharing.cli.main import app
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "eval" in command_names

    def test_main_app_loads(self):
        from critic_sharing.cli.main import app
        assert app is not None


class TestShimsRemoved:
    def test_no_envs_push_import(self):
        with pytest.raises(ImportError):
            import critic_sharing.envs.push  # noqa: F401

    def test_no_envs_nav_import(self):
        with pytest.raises(ImportError):
            import critic_sharing.envs.nav  # noqa: F401

    def test_canonical_push_import(self):
        from critic_sharing.scenarios.obstacle_arena.push_env import PushObstacleEnv
        assert PushObstacleEnv is not None

    def test_canonical_nav_import(self):
        from critic_sharing.scenarios.obstacle_arena.nav_env import OneObstacleReachEnv
        assert OneObstacleReachEnv is not None
