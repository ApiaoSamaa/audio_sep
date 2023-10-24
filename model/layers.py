import torch
from typing import Any, Dict, Optional
import torch
from torch import nn
from torch.nn import functional as F
import math

class ScaledEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.0, smooth: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # when summing gaussian, scale raises as sqrt(n), so we normalize by that.
            weight = weight / torch.arange(1, num_embeddings + 1).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embedding(x) * self.scale
        return out

class HEncLayer(torch.nn.Module):
    def __init__(
        self,
        chin: int,
        chout: int,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 4,
        empty: bool = False,
        freq: bool = True,
        norm_type: str = "group_norm",
        context: int = 0,
        dconv_kw: Optional[Dict[str, Any]] = None,
        pad: bool = True,
    ):
        super().__init__()
        if dconv_kw is None:
            dconv_kw = {}
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm_type == "group_norm":
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        pad_val = kernel_size // 4 if pad else 0
        klass = nn.Conv1d
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.pad = pad_val
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad_val = [pad_val, 0]
            klass = nn.Conv2d
        self.conv = klass(chin, chout, kernel_size, stride, pad_val)
        self.norm1 = norm_fn(chout)

        if self.empty:
            self.rewrite = nn.Identity()
            self.norm2 = nn.Identity()
            self.dconv = nn.Identity()
        else:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)
            self.dconv = _DConv(chout, **dconv_kw)

    def forward(self, x: torch.Tensor, inject: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            if inject.shape[-1] != y.shape[-1]:
                raise ValueError("Injection shapes do not align")
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.freq:
            B, C, Fr, T = y.shape
            y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = self.dconv(y)
        z = self.norm2(self.rewrite(y))
        z = F.glu(z, dim=1)
        return z

class HDecLayer(torch.nn.Module):
    def __init__(
        self,
        chin: int,
        chout: int,
        last: bool = False,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 1,
        empty: bool = False,
        freq: bool = True,
        norm_type: str = "group_norm",
        context: int = 1,
        dconv_kw: Optional[Dict[str, Any]] = None,
        pad: bool = True,
    ):
        super().__init__()
        if dconv_kw is None:
            dconv_kw = {}
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm_type == "group_norm":
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            if (kernel_size - stride) % 2 != 0:
                raise ValueError("Kernel size and stride do not align")
            pad = (kernel_size - stride) // 2
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            self.rewrite = nn.Identity()
            self.norm1 = nn.Identity()
        else:
            self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            self.norm1 = norm_fn(2 * chin)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor], length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip
            y = F.glu(self.norm1(self.rewrite(x)), dim=1)
        else:
            y = x
            if skip is not None:
                raise ValueError("Skip must be none when empty is true.")

        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad : -self.pad, :]
        else:
            z = z[..., self.pad : self.pad + length]
            if z.shape[-1] != length:
                raise ValueError("Last index of z must be equal to length")
        if not self.last:
            z = F.gelu(z)

        return z, y

class _DConv(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        compress: float = 4,
        depth: int = 2,
        init: float = 1e-4,
        norm_type: str = "group_norm",
        attn: bool = False,
        heads: int = 4,
        ndecay: int = 4,
        lstm: bool = False,
        kernel_size: int = 3,
    ):

        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size should not be divisible by 2")
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        norm_fn: tp.Callable[[int], nn.Module]
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm_type == "group_norm":
            norm_fn = lambda d: nn.GroupNorm(1, d)  # noqa

        hidden = int(channels / compress)

        act = nn.GELU

        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            dilation = pow(2, d) if dilate else 1
            padding = dilation * (kernel_size // 2)
            mods = [
                nn.Conv1d(channels, hidden, kernel_size, dilation=dilation, padding=padding),
                norm_fn(hidden),
                act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels),
                nn.GLU(1),
                _LayerScale(channels, init),
            ]
            if attn:
                mods.insert(3, _LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, _BLSTM(hidden, layers=2, skip=True))
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class _BLSTM(torch.nn.Module):
    def __init__(self, dim, layers: int = 1, skip: bool = False):
        super().__init__()
        self.max_steps = 200
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        y = x
        framed = False
        width = 0
        stride = 0
        nframes = 0
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = _unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)

        x = x.permute(2, 0, 1)

        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out
        if self.skip:
            x = x + y

        return x
def _unfold(a: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    shape = list(a.shape[:-1])
    length = int(a.shape[-1])
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(input=a, pad=[0, tgt_length - length])
    strides = [a.stride(dim) for dim in range(a.dim())]
    if strides[-1] != 1:
        raise ValueError("Data should be contiguous.")
    strides = strides[:-1] + [stride, 1]
    shape.append(n_frames)
    shape.append(kernel_size)
    return a.as_strided(shape, strides)

class _LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0):
        r"""
        Args:
            channels (int): Size of  rescaling
            init (float, optional): Scale to default to (default: 0)
        """
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""LayerScale forward call

        Args:
            x (torch.Tensor): input tensor for LayerScale

        Returns:
            Tensor
                Output after rescaling tensor.
        """
        return self.scale[:, None] * x

class _LocalState(nn.Module):
    def __init__(self, channels: int, heads: int = 4, ndecay: int = 4):
        super(_LocalState, self).__init__()
        if channels % heads != 0:
            raise ValueError("Channels must be divisible by heads.")
        self.heads = heads
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)

        self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
        if ndecay:
            # Initialize decay close to zero (there is a sigmoid), for maximum initial window.
            self.query_decay.weight.data *= 0.01
            if self.query_decay.bias is None:
                raise ValueError("bias must not be None.")
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * 0, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        # left index are keys, right index are queries
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        # t are keys, s are queries
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= math.sqrt(keys.shape[2])
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = -decays.view(-1, 1, 1) * delta.abs() / math.sqrt(self.ndecay)
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Kill self reference.
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)

        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)
