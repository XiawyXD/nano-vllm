"""CPU元数据回归：auto白名单/人工门槛/默认回退，不启动CUDA。"""
import torch
from nanovllm.layers.tensorcore_mlp import should_use_fused_swiglu


class TensorMetadata:
    """只模拟选择器读取的tensor元信息，不模拟GPU计算正确性。"""
    def __init__(self, shape, dtype=torch.bfloat16, contiguous=True, cuda=True):
        self.shape, self.ndim = shape, len(shape)
        self.dtype, self.is_cuda = dtype, cuda
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous


def main():
    """端到端负收益后auto白名单为空；显式实验门槛和默认Torch均不变。"""
    count = 0
    weight = TensorMetadata((12288, 2048))
    for m in (0, 1, 2, 4, 8, 16, 17, 32, 33, 128, 512):
        x = TensorMetadata((m, 2048))
        assert not should_use_fused_swiglu(x, weight, 'auto', 128, (8, 9))
        assert not should_use_fused_swiglu(x, weight, 'torch', 1, (8, 9))
        assert should_use_fused_swiglu(x, weight, 'triton', 128, (8, 9)) == (m >= 128)
        count += 3
    x = TensorMetadata((16, 2048))
    for capability in (None, (8, 0), (9, 0)):
        assert not should_use_fused_swiglu(x, weight, 'auto', 1, capability)
        count += 1
    for invalid_x, invalid_weight in (
        (TensorMetadata((16, 2048), dtype=torch.float16), weight),
        (TensorMetadata((16, 2048), contiguous=False), weight),
        (TensorMetadata((16, 2048), cuda=False), weight),
        (TensorMetadata((16, 1024)), TensorMetadata((6144, 1024))),
        (x, TensorMetadata((6144, 2048))),  # TP2本地I，不在本轮实测白名单。
        (x, TensorMetadata((12288, 2048), contiguous=False)),
        (TensorMetadata((1, 16, 2048)), weight),
    ):
        assert not should_use_fused_swiglu(invalid_x, invalid_weight, 'auto', 1, (8, 9))
        count += 1
    try:
        should_use_fused_swiglu(x, weight, 'unknown', 1, (8, 9))
    except ValueError:
        count += 1
    else:
        raise AssertionError('unknown backend accepted')
    print(f'PASS: {count} CPU MLP dispatch assertions; no GPU kernel tested')


if __name__ == '__main__':
    main()
