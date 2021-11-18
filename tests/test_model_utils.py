# -*- coding: utf-8 -*-
"""
Created on 11 Jan 2021 14:55:01
@author: jiahuei
"""
import unittest
from sparse_caption.utils import model_utils


class TestModelUtils(unittest.TestCase):

    # noinspection PyTypeChecker
    def test_map_recursive(self):
        def fn(_):
            return _ ** 2

        with self.subTest("Sub-test: dict"):
            x = dict(a=9, b=-1.2, c=dict(x=7.2, y=[-2, 3]), d=[], e=dict())
            x = model_utils.map_recursive(x, fn)
            self.assertEqual(x, {"a": 81, "b": 1.44, "c": {"x": 51.84, "y": [4, 9]}, "d": [], "e": {}})
        with self.subTest("Sub-test: tuple"):
            x = (9, -1.2, dict(x=7.2, y=[-2, 3]), [], dict())
            x = model_utils.map_recursive(x, fn)
            self.assertEqual(x, (81, 1.44, {"x": 51.84, "y": [4, 9]}, [], {}))
        with self.subTest("Sub-test: list"):
            x = [9, -1.2, dict(x=7.2, y=[-2, 3]), [], dict()]
            x = model_utils.map_recursive(x, fn)
            self.assertEqual(x, [81, 1.44, {"x": 51.84, "y": [4, 9]}, [], {}])
        with self.subTest("Sub-test: assert TypeError"):
            x = [9, -1.2, dict(x=7.2, y=[-2, 3]), None, dict()]
            self.assertRaises(TypeError, lambda: model_utils.map_recursive(x, fn))


if __name__ == "__main__":
    unittest.main()
