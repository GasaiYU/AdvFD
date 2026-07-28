import unittest

import torch

from frechet_distance.adversarial import shared_feature_norm_offset_penalty


class SharedFeatureNormOffsetPenaltyTest(unittest.TestCase):
    def test_matching_features_have_zero_penalty(self):
        real = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        fake = torch.tensor([[2.0, 1.0], [4.0, 3.0]])

        penalty, metrics = shared_feature_norm_offset_penalty(
            real,
            real,
            fake,
            fake,
        )

        self.assertEqual(float(metrics["second_moment_ratio"]), 1.0)
        self.assertEqual(float(penalty), 0.0)

    def test_common_scaling_matches_closed_form_and_only_adv_gets_grad(self):
        ref_real = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )
        ref_fake = torch.tensor(
            [[2.0, 1.0], [4.0, 3.0]],
            requires_grad=True,
        )
        adv_real = (2.0 * ref_real.detach()).requires_grad_()
        adv_fake = (2.0 * ref_fake.detach()).requires_grad_()

        penalty, metrics = shared_feature_norm_offset_penalty(
            adv_real,
            ref_real,
            adv_fake,
            ref_fake,
        )
        penalty.backward()

        self.assertAlmostEqual(float(metrics["second_moment_ratio"]), 4.0)
        self.assertAlmostEqual(float(penalty.detach()), 9.0)
        self.assertGreater(float(adv_real.grad.abs().sum()), 0.0)
        self.assertGreater(float(adv_fake.grad.abs().sum()), 0.0)
        self.assertIsNone(ref_real.grad)
        self.assertIsNone(ref_fake.grad)

    def test_low_precision_inputs_accumulate_in_fp32(self):
        ref_real = torch.ones(2, 3, dtype=torch.float16)
        ref_fake = torch.full((2, 3), 2.0, dtype=torch.float16)

        penalty, metrics = shared_feature_norm_offset_penalty(
            ref_real,
            ref_real,
            ref_fake,
            ref_fake,
        )

        self.assertEqual(penalty.dtype, torch.float32)
        self.assertEqual(metrics["adv_rms"].dtype, torch.float32)

    def test_zero_reference_second_moment_is_rejected(self):
        adv = torch.ones(2, 3)
        ref = torch.zeros(2, 3)

        with self.assertRaisesRegex(ValueError, "must be > 0"):
            shared_feature_norm_offset_penalty(adv, ref, adv, ref)


if __name__ == "__main__":
    unittest.main()
