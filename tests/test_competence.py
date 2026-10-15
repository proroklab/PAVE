"""The standalone competence protocol (critic_sharing.training.competence)."""

from critic_sharing.training.competence import success_rate


class _Env:
    """Episode k lasts 3 steps; the success event fires at step 2 iff k is even."""
    SUCCESS_INFO_KEY = "done_it"

    def reset(self, seed):
        self.k, self.t = seed, 0
        return 0, {}

    def step(self, action):
        self.t += 1
        info = {"done_it": self.t == 2 and self.k % 2 == 0}
        return 0, 0.0, self.t == 3, False, info


class _Model:
    def predict(self, obs, deterministic):
        assert deterministic
        return 0, None


def test_success_counts_an_event_at_any_step():
    # The event fires mid-episode, not on the last step: still a success.
    assert success_rate(_Model(), _Env(), n_episodes=4, base_seed=0) == 0.5


def test_reset_seeds_start_at_base_seed():
    # base_seed 1 -> seeds 1, 2, 3: only seed 2 succeeds.
    assert success_rate(_Model(), _Env(), n_episodes=3, base_seed=1) == 1 / 3
