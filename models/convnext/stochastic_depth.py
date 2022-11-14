# MIT License Copyright (c) 2022 joh-fischer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        assert 0. <= drop_prob <= 1., f"Drop probability must be between 0 and 1, got {drop_prob}"

        self.drop_prob = drop_prob
        self.survival_prob = 1. - self.drop_prob

    def forward(self, x: torch.Tensor):
        if not self.training:
            return x

        if self.drop_prob == 0.:
            return x

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        survive = x.new_empty(shape).bernoulli_(self.survival_prob)

        if self.survival_prob > 0.:
           survive.div_(self.survival_prob)

        return x * survive


if __name__ == "__main__":
    ipt = torch.randint(0, 10, (10, 1, 4, 4)).float()

    dp = DropPath(0.3)
    out = dp(ipt)

    print("ipt:", ipt.shape)  # torch.Size([10, 3, 4, 4])
    print("out:", out.shape)  # torch.Size([10, 3, 4, 4])
    print(out)