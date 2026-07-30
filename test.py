import torch

def row_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    给定形状为 (N, D) 的浮点 Tensor，将每一行除以其 L2 范数。

    对于范数小于 eps 的行，返回全零行。

    要求：
    - 返回 Tensor 与 x 具有相同形状、dtype 和 device。
    - 不允许逐行使用 Python 循环。

    示例：
    >>> x = torch.tensor([[3., 4.], [0., 0.]])
    >>> row_normalize(x)
    tensor([[0.6000, 0.8000],
            [0.0000, 0.0000]])
    """
    norm_x = torch.linalg.norm(x)

    return torch.where(
        x >= eps,
        x / norm_x,
        torch.zeros_like(x)
    )

if __name__ == "__main__":
    x = torch.tensor([[3., 4.], [0., 0.]])
    a = row_normalize(x)
    print(a)