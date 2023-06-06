from collections import UserDict


class Dictionary(UserDict):
    def __init__(self, dict=None, default_item=None, /, **kwargs):
        super().__init__(self, dict=dict)
        self.default_item = default_item
    def __getitem__(self, item):
        if item in super().keys():
            return super().__getitem__(item)
        else:
            return self.default_item