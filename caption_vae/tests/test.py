# -*- coding: utf-8 -*-
"""
Created on 10 Sep 2020 19:07:16
@author: jiahuei
"""


class A:
    def __init__(self):
        print(f"__init__ base: {self.__class__.__name__}")

    @classmethod
    def init(cls):
        print(f"init base: {cls.__name__}")
        return cls()


class B(A):
    def __init__(self):
        super().__init__()
        print(self.__class__.__name__)

    @classmethod
    def init(cls):
        return super().init()


class C(A):
    def __init__(self):
        super().__init__()
        print(self.__class__.__name__)

    @classmethod
    def init(cls):
        return cls()


a = A.init()
print()
b = B.init()
print()
c = C.init()
print()
