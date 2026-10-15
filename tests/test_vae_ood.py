"""Tests for the VAE/OOD pipeline.

Three thin slices:

1. ``percentile_calibration`` math — pure-numpy, no torch needed.
2. ``JointUtilityCost`` trust-weighted form — verifies the cost reduces
   to the un-OOD baseline when ``ood_metric is None`` and applies the
   ``alpha * (1 - score)`` trust factor when wired.
3. End-to-end smoke: small synthetic 2-D dataset → ``VAE`` →
   ``percentile_calibration`` → ``OODMetric.metric`` returns the
   expected shape and is bounded in ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np

from critic_sharing.planning.costs import JointUtilityCost


# ---------------------------------------------------------------------------
# Percentile calibration
# ---------------------------------------------------------------------------


class TestPercentileCalibration:
    def test_threshold_below_max_loss(self):
        from critic_sharing.training.vae_from_rollouts import percentile_calibration

        losses = np.linspace(0.0, 10.0, 100, dtype=np.float32)
        cal = percentile_calibration(losses, threshold_pct=95.0, max_loss_pct=99.9)
        assert cal.method == "percentile"
        assert cal.threshold < cal.max_loss

    def test_degenerate_constant_losses(self):
        """When all training losses are identical, percentile_calibration must
        still produce a non-zero (max_loss - threshold) span so the OOD
        score normalisation doesn't divide by ~zero."""
        from critic_sharing.training.vae_from_rollouts import percentile_calibration

        losses = np.full(100, 0.5, dtype=np.float32)
        cal = percentile_calibration(losses)
        assert cal.max_loss > cal.threshold


# ---------------------------------------------------------------------------
# JointUtilityCost trust-weighted form
# ---------------------------------------------------------------------------


def _make_cost(ood_metric=None, alpha=1.0):
    def fake_v(obs):
        return np.full(obs.shape[0], 5.0, dtype=np.float32)

    return JointUtilityCost(
        follower_value_fn=fake_v,
        leader_gamma=0.99,
        utility_type="sum",
        leader_range=(0.0, 30.0),
        follower_range=(0.0, 10.0),
        ood_metric=ood_metric,
        alpha=alpha,
    )


def _eval(cost, n=4, h=5):
    obs = np.zeros((n, h + 1, 3), dtype=np.float32)
    actions = np.zeros((n, h, 1), dtype=np.float32)
    rewards = np.ones((n, h), dtype=np.float32)
    return cost.evaluate(obs, actions, rewards)


class TestTrustWeightedCost:
    def test_alpha_applies_without_a_detector(self):
        """``alpha`` scales V_F even when ``ood_metric is None``.

        This is the behaviour introduced by the ood_lambda -> alpha rename:
        the value factor is applied regardless of whether a detector is
        wired in, so alpha is NOT ignored without one. (The pre-rename
        ``ood_lambda`` was a penalty weight and did nothing without a
        detector, which is why this test used to assert equality.)

        F_norm = V / follower_range = 5 / 10 = 0.5, and cost is the negated
        utility, so going alpha 1 -> 60 shifts cost by -(59 * 0.5).
        """
        c_alpha1 = _make_cost(ood_metric=None, alpha=1.0)
        c_alpha60 = _make_cost(ood_metric=None, alpha=60.0)
        np.testing.assert_array_almost_equal(
            _eval(c_alpha60) - _eval(c_alpha1), -29.5, decimal=5,
        )

    def test_high_ood_silences_v_f(self):
        """OOD score 1.0 → trust 0 → V_F contribution silenced."""
        def ood_max(obs):
            return np.full(obs.shape[0], 1.0, dtype=np.float32)

        c_no_ood = _make_cost(ood_metric=None, alpha=1.0)
        c_max_ood = _make_cost(ood_metric=ood_max, alpha=1.0)
        # max_ood_cost = -L_norm (no V_F contribution).
        # no_ood_cost  = -(L_norm + F_norm).
        # Therefore max_ood_cost > no_ood_cost (less negative).
        assert (_eval(c_max_ood) > _eval(c_no_ood)).all()

    def test_zero_ood_matches_baseline(self):
        """OOD score 0 → trust 1 → with alpha=1, identical to no-OOD baseline."""
        def ood_zero(obs):
            return np.zeros(obs.shape[0], dtype=np.float32)

        c_baseline = _make_cost(ood_metric=None, alpha=1.0)
        c_zero_ood = _make_cost(ood_metric=ood_zero, alpha=1.0)
        np.testing.assert_array_almost_equal(_eval(c_baseline), _eval(c_zero_ood))

    def test_partial_ood_interpolates(self):
        """OOD = 0.5 should give half-strength V_F contribution."""
        def ood_half(obs):
            return np.full(obs.shape[0], 0.5, dtype=np.float32)
        def ood_zero(obs):
            return np.zeros(obs.shape[0], dtype=np.float32)

        c_full = _make_cost(ood_metric=ood_zero, alpha=1.0)
        c_half = _make_cost(ood_metric=ood_half, alpha=1.0)
        c_off = _make_cost(ood_metric=lambda o: np.ones(o.shape[0], np.float32),
                           alpha=1.0)

        # Costs (negated utility) — higher = worse.
        # off (trust=0) > half (trust=0.5) > full (trust=1).
        assert (_eval(c_off) > _eval(c_half)).all()
        assert (_eval(c_half) > _eval(c_full)).all()

    def test_alpha_scales_v_f(self):
        """alpha=2 with trust=1 doubles V_F's contribution vs alpha=1."""
        def ood_zero(obs):
            return np.zeros(obs.shape[0], dtype=np.float32)

        c_alpha1 = _make_cost(ood_metric=ood_zero, alpha=1.0)
        c_alpha2 = _make_cost(ood_metric=ood_zero, alpha=2.0)
        cost1 = _eval(c_alpha1)
        cost2 = _eval(c_alpha2)
        # cost1 = -(L + F_norm), cost2 = -(L + 2*F_norm).
        # So cost2 - cost1 should equal -F_norm = -0.5 (V=5/range=10).
        np.testing.assert_array_almost_equal(cost2 - cost1, -0.5, decimal=5)


# ---------------------------------------------------------------------------
# End-to-end VAE smoke
# ---------------------------------------------------------------------------


class TestVaeSmoke:
    def test_train_score_pipeline(self):
        """Train a tiny VAE on synthetic 2-D Gaussian samples; confirm
        OOD scores are bounded in [0, 1] and that out-of-distribution
        points score noticeably higher than in-distribution points."""
        import torch as th

        from critic_sharing.guidance.ood import OODMetric, reconstruction_losses
        from critic_sharing.models.vae import VAE
        from critic_sharing.training.vae import train_vae
        from critic_sharing.training.vae_from_rollouts import percentile_calibration

        rng = np.random.default_rng(0)
        in_dist = rng.standard_normal(size=(500, 2)).astype(np.float32)
        # Far-OOD samples, an order of magnitude away.
        ood = (rng.standard_normal(size=(50, 2)) + 20.0).astype(np.float32)

        vae = VAE(input_dim=2, latent_dim=2)
        loader = th.utils.data.DataLoader(
            th.utils.data.TensorDataset(th.from_numpy(in_dist)),
            batch_size=32, shuffle=True,
        )
        train_vae(
            model=vae, dataloader=loader,
            optimizer=th.optim.Adam(vae.parameters(), lr=1e-2),
            num_epochs=15,
        )

        train_losses = reconstruction_losses(vae, in_dist)
        cal = percentile_calibration(train_losses)
        metric = OODMetric(vae, threshold=cal.threshold, max_loss=cal.max_loss)

        in_scores = metric.metric(in_dist[:50])
        ood_scores = metric.metric(ood)

        assert in_scores.min() >= 0.0 and in_scores.max() <= 1.0
        assert ood_scores.min() >= 0.0 and ood_scores.max() <= 1.0
        # Mean OOD score should be substantially higher than in-dist.
        assert ood_scores.mean() > in_scores.mean() + 0.3, (
            "OOD samples didn't score noticeably higher than in-dist: "
            "in={i:.3f} ood={o:.3f}".format(
                i=float(in_scores.mean()), o=float(ood_scores.mean()),
            )
        )
