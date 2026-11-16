"""Check retained visual history and natural generation termination."""
import sys
import unittest
from types import SimpleNamespace
from run_inference import observed_frame_indices, configure_generation, generation_limits
from run_proassist import generate_until_eos


class ContextAlignmentTests(unittest.TestCase):
    def test_fractional_fps_keeps_complete_prefix(self):
        indices = observed_frame_indices(0, 60, 29.97, 3000, 2)
        self.assertEqual(len(indices), 121)
        self.assertEqual(indices[:4], [0, 14, 29, 44])
        self.assertEqual(indices[-1], 1798)
        self.assertGreater(len(indices), 32)
        self.assertEqual(len(observed_frame_indices(0, 60, 29.97, 3000, 2, 32)), 32)
        self.assertTrue(all(i >= 300 for i in observed_frame_indices(10, 11, 30, 1000, 2)))

    def test_clear_inherited_generation_budget(self):
        cfg = SimpleNamespace(max_new_tokens=256, max_length=20, max_time=1,
                              min_new_tokens=50, forced_eos_token_id=2)
        configure_generation(cfg, 0)
        self.assertIsNone(cfg.max_new_tokens)
        self.assertIsNone(cfg.max_time)
        self.assertIsNone(cfg.forced_eos_token_id)
        self.assertIsNone(cfg.min_new_tokens)
        self.assertEqual(cfg.max_length, sys.maxsize)
        self.assertIsNone(generation_limits(0)['max_new_tokens'])
        self.assertEqual(generation_limits(64)['max_new_tokens'], 64)

    def test_proassist_dynamic_decode_exceeds_old_budget_and_retains_cache(self):
        import torch
        calls = []
        def forward(**kw):
            calls.append(kw['past_key_values'])
            logits = torch.zeros(1, 1, 4)
            logits[..., 2 if len(calls) == 301 else 1] = 2
            return SimpleNamespace(logits=logits, past_key_values=len(calls),
                                   w2t_probs=torch.tensor([[.1]]))
        model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2, w2t_target_id=3),
                                forward=forward, get_input_embeddings=lambda: lambda x: x)
        output, cache = generate_until_eos(model, torch.zeros(1, 1, 1), None, .3)
        self.assertEqual(output.shape[1], 301)
        self.assertEqual(output[0, -1].item(), 2)
        self.assertEqual(cache, 301)
        self.assertEqual(calls[1], 1)
        model.forward = lambda **kw: SimpleNamespace(logits=torch.zeros(1, 1, 4),
            past_key_values='silent-cache', w2t_probs=torch.tensor([[.9]]))
        output, cache = generate_until_eos(model, None, None, .3)
        self.assertEqual(output.tolist(), [[3]])
        self.assertEqual(cache, 'silent-cache')


if __name__ == '__main__':
    unittest.main()
