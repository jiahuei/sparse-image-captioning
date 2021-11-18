# -*- coding: utf-8 -*-
"""
Created on 06 Jan 2021 21:29:39
@author: jiahuei

python -m unittest pruning/test_prune.py
"""
import unittest
import torch
from copy import deepcopy
from torch import nn, optim
from torch.nn import functional as F
from sparse_caption.pruning import prune
from sparse_caption.pruning.masked_layer import MaskedLinear, MaskedLSTMCell, MaskedEmbedding
from sparse_caption.utils.model_utils import set_seed, map_to_cuda


# noinspection PyAbstractClass
class Model(prune.PruningMixin, nn.Module):
    def __init__(self, mask_type):
        super().__init__(mask_type=mask_type, mask_freeze_scope="out.")
        mask_params = {"mask_type": mask_type, "mask_init_value": 5.0}
        self.embed = MaskedEmbedding(3, 4, **mask_params)
        self.lstm = MaskedLSTMCell(4, 4, **mask_params)
        ffl = MaskedLinear(4, 4, **mask_params)
        self.ff = nn.Sequential(*(deepcopy(ffl) for _ in range(2)))
        self.out = MaskedLinear(4, 3, **mask_params)
        self.loss_layer = nn.CrossEntropyLoss()
        map_to_cuda(self)

    def forward(self, inputs):
        inputs = map_to_cuda(inputs)
        net = F.relu(self.embed(inputs))
        net = F.relu(self.lstm(net)[0])
        for ly in self.ff:
            net = F.dropout(F.relu(ly(net)), p=0.15, training=self.training)
        return self.out(net)

    def compute_loss(self, predictions, labels, sparsity_target=None, step=None, max_step=None):
        labels = map_to_cuda(labels)
        loss = self.loss_layer(predictions, labels)
        if self.mask_type == prune.REGULAR:
            assert sparsity_target is not None
            assert step is not None
            assert max_step is not None
            loss += self.compute_sparsity_loss(sparsity_target, weight=120, current_step=step, max_step=max_step)
        return loss

    def get_optimizer(self):
        optim_params = [{"params": list(self.all_weights(named=False))}]
        if self.mask_type in prune.SUPER_MASKS:
            optim_params += [
                {
                    "params": list(self.active_pruning_masks(named=False)),
                    "lr": 10,
                    "weight_decay": 0,
                }
            ]
        optimizer = optim.Adam(optim_params, lr=1e-2, weight_decay=1e-5)
        return optimizer

    def train_self(self, sparsity_target, iters=40):
        optimizer = self.get_optimizer()
        for i in range(iters):
            inputs, labels = self.get_inputs_and_labels()
            optimizer.zero_grad()
            loss = self.compute_loss(self(inputs), labels, sparsity_target, i, iters)
            loss.backward()
            optimizer.step()
            if self.mask_type in prune.MAG_ANNEAL:
                self.update_masks_gradual(
                    sparsity_target=sparsity_target,
                    current_step=i,
                    start_step=10,
                    prune_steps=max(1, int(iters / 4)),
                    prune_frequency=3,
                )

    def get_sparsity(self, active=True):
        if active:
            return float(self.active_mask_sparsities[0].item())
        else:
            return float(self.all_mask_sparsities[0].item())

    def get_weight_sparsity(self):
        return float(self.all_weight_sparsities[0].item())

    @staticmethod
    def get_inputs_and_labels():
        inputs = torch.randint(low=0, high=2, size=(5,), dtype=torch.int64)
        labels = torch.randint(low=0, high=2, size=(5,), dtype=torch.int64)
        return inputs, labels


class TestPrune(unittest.TestCase):
    SPARSITY_TARGET = 0.8

    def setUp(self) -> None:
        set_seed(8888)

    def _test_model(self, mask_type):
        model = Model(mask_type)
        with self.subTest(f"{mask_type} : Initial sparsity check"):
            self.assertEqual(model.get_sparsity(active=False), 0, "Initial sparsity should be zero")
        if model.mask_type in prune.MAG_HARD + prune.LOTTERY + [prune.SNIP]:
            if model.mask_type == prune.SNIP:
                inputs, labels = model.get_inputs_and_labels()
                loss = model.compute_loss(model(inputs), labels)
                loss.backward()
            model.update_masks_once(sparsity_target=self.SPARSITY_TARGET)
            with self.subTest(f"{mask_type} : One-shot pruning sparsity check"):
                self.assertAlmostEqual(
                    model.get_sparsity(active=True),
                    self.SPARSITY_TARGET,
                    delta=0.05,
                    msg=f"Sparsity should be {self.SPARSITY_TARGET}",
                )
        model.train_self(self.SPARSITY_TARGET)
        with self.subTest(f"{mask_type} : Final sparsity check"):
            self.assertAlmostEqual(
                model.get_sparsity(active=True),
                self.SPARSITY_TARGET,
                delta=0.3 if mask_type == prune.REGULAR else 0.05,
                msg=f"Sparsity should be {self.SPARSITY_TARGET}",
            )
        with self.subTest(f"{mask_type} : Active sparsity check"):
            self.assertGreater(
                model.get_sparsity(active=True),
                model.get_sparsity(active=False),
                "Active sparsity should be higher as `out` layer is not pruned.",
            )
        with self.subTest(f"{mask_type} : Weight sparsity check"):
            self.assertEqual(model.get_weight_sparsity(), 0, "Weights should not be pruned yet.")
        model.prune_weights()
        with self.subTest(f"{mask_type} : Weight sparsity check"):
            self.assertAlmostEqual(
                model.get_weight_sparsity(),
                self.SPARSITY_TARGET,
                delta=0.3,
                msg=f"Weight sparsity after pruning should be {self.SPARSITY_TARGET}",
            )

    def test_prune(self):
        for mask_type in (
            prune.REGULAR,
            prune.MAG_BLIND,
            prune.MAG_DIST,
            prune.MAG_UNIFORM,
            prune.SNIP,
            prune.MAG_GRAD_BLIND,
            prune.MAG_GRAD_UNIFORM,
            # prune.MAG_GRAD_DIST,
            prune.LOTTERY_MAG_BLIND,
            prune.LOTTERY_MAG_UNIFORM,
            prune.LOTTERY_MAG_DIST,
        ):
            sub_test = f"Testing mask_type = {mask_type}"
            with self.subTest(sub_test):
                print(sub_test)
                self._test_model(mask_type)


if __name__ == "__main__":
    unittest.main()
