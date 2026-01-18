"""
All of the functions in this file are taken from the official SReC github page - 
https://github.com/caoscott/SReC/tree/master

Small modification are done by the authors of CLIDE papers.
"""
import torch
import torch.nn as nn
import numpy as np
import math
from collections import defaultdict
from torch.nn import functional as F
from torch.utils import data
from typing import DefaultDict, Generator, KeysView, List, NamedTuple, Tuple, Union
import PIL.Image as Image


class configs:
    resblocks = 3
    n_feats = 64
    scale = 3
    K = 10
    log_likelihood = True
    _NUM_PARAMS_RGB = 4  
    _NUM_PARAMS_OTHER = 3
    _LOG_SCALES_MIN = -7.


class CDFOut(NamedTuple):
    logit_probs_c_sm: torch.Tensor
    means_c: torch.Tensor
    log_scales_c: torch.Tensor
    K: int
    targets: torch.Tensor


class StackedAtrousConvs(nn.Module):
    def __init__(self,
                 atrous_rates_str: Union[str, int],
                 Cin: int,
                 Cout: int,
                 bias: bool = True,
                 kernel_size: int = 3) -> None:
        super(StackedAtrousConvs, self).__init__()
        atrous_rates = self._parse_atrous_rates_str(atrous_rates_str)
        self.atrous = nn.ModuleList(
            [conv(Cin, Cin, kernel_size, rate=rate)
             for rate in atrous_rates])
        self.lin = conv(len(atrous_rates) * Cin, Cout, 1, bias=bias)
        self._extra_repr = 'rates={}'.format(atrous_rates)

    @staticmethod
    def _parse_atrous_rates_str(atrous_rates_str: Union[str, int]) -> List[int]:
        # expected to either be an int or a comma-separated string 1,2,4
        if isinstance(atrous_rates_str, int):
            return [atrous_rates_str]
        else:
            return list(map(int, atrous_rates_str.split(',')))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = torch.cat([atrous(x)
                       for atrous in self.atrous], dim=1)  # type: ignore
        x = self.lin(x)
        return x


class AtrousProbabilityClassifier(nn.Module):
    def __init__(self,
                 in_ch: int,
                 C: int,
                 num_params: int,
                 K: int = 10,
                 kernel_size: int = 3,
                 atrous_rates_str: str = '1,2,4') -> None:
        super(AtrousProbabilityClassifier, self).__init__()

        Kp = non_shared_get_Kp(K, C, num_params)

        self.atrous = StackedAtrousConvs(atrous_rates_str, in_ch, Kp,
                                         kernel_size=kernel_size)
        self._repr = f'C={C}; K={K}; Kp={Kp}; rates={atrous_rates_str}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        :param x: N C H W
        :return: N Kp H W
        """
        return self.atrous(x)


class DiscretizedMixLogisticLoss(nn.Module):
    def __init__(self, rgb_scale: bool, x_min=0, x_max=255, L=256):
        
        super(DiscretizedMixLogisticLoss, self).__init__()
        self.rgb_scale = rgb_scale
        self.x_min = x_min
        self.x_max = x_max
        self.L = L
        self.use_coeffs = rgb_scale
        self._num_params = (
            configs._NUM_PARAMS_RGB if rgb_scale else
            configs._NUM_PARAMS_OTHER)

        self._nonshared_coeffs_act = torch.sigmoid

        self.bin_width = (x_max - x_min) / (L-1)
        self.x_lower_bound = x_min + 0.001
        self.x_upper_bound = x_max - 0.001

        self._extra_repr = 'DMLL: x={}, L={}, coeffs={}, P={}, bin_width={}'.format(
            (self.x_min, self.x_max), self.L, self.use_coeffs, self._num_params, self.bin_width)

    def log_cdf(self, lo, hi, means, log_scales):
        assert torch.all(lo <= hi), f"{lo[lo > hi]} > {hi[lo > hi]}"
        assert lo.min() >= self.x_min and hi.max() <= self.x_max, \
            '{},{} not in {},{}'.format(
                lo.min(), hi.max(), self.x_min, self.x_max)

        centered_lo = lo - means  # NCKHW
        centered_hi = hi - means

        inv_stdv = torch.exp(-log_scales)
        normalized_lo = inv_stdv * (
            centered_lo - self.bin_width/2)  # sigma' * (x - mu - 1/255)
        lo_cond = (lo >= self.x_lower_bound).float()
        cdf_lo = lo_cond * torch.sigmoid(normalized_lo)
        normalized_hi = inv_stdv * (centered_hi + self.bin_width/2)
        hi_cond = (hi <= self.x_upper_bound).float()
        cdf_hi = hi_cond * torch.sigmoid(normalized_hi) + (1 - hi_cond)  # * 1.
        cdf_delta = cdf_hi - cdf_lo
        log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=1e-12))

        assert not torch.any(
            log_cdf_delta > 1e-6
        ), f"{log_cdf_delta[log_cdf_delta > 1e-6]}"
        return log_cdf_delta

    def forward(  
            self, x: torch.Tensor, l: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: labels, i.e., NCHW, float
        :param l: predicted distribution, i.e., NKpHW, see above
        :return: log-likelihood, as NHW if shared, NCHW if non_shared pis
        """
        assert x.min() >= self.x_min and x.max() <= self.x_max, \
            f'{x.min()},{x.max()} not in {self.x_min},{self.x_max}'

        x, logit_pis, means, log_scales, _ = self._extract_non_shared(x, l)
        log_probs = self.log_cdf(x, x, means, log_scales)

        log_weights = F.log_softmax(logit_pis, dim=2)
        log_probs_weighted = log_weights + log_probs

        nll = -torch.logsumexp(log_probs_weighted, dim=2)

        probs_weighted = torch.exp(log_probs_weighted)
        entropy = -torch.sum(probs_weighted * log_probs_weighted, dim=2)

        return nll, entropy

    def _extract_non_shared(self, x, l):
        """
        :param x: targets, NCHW
        :param l: output of net, NKpHW, see above
        :return:
            x NC1HW,
            logit_probs NCKHW (probabilites of scales, i.e., \pi_k)
            means NCKHW,
            log_scales NCKHW (variances),
            K (number of mixtures)
        """
        N, C, H, W = x.shape
        Kp = l.shape[1]

        K = non_shared_get_K(Kp, C, self._num_params)

        # we have, for each channel: K pi / K mu / K sigma / [K coeffs]
        # note that this only holds for C=3 as for other channels,
        # there would be more than 3*K coeffs
        # but non_shared only holds for the C=3 case
        l = l.reshape(N, self._num_params, C, K, H, W)

        logit_probs = l[:, 0, ...]  # NCKHW
        means = l[:, 1, ...]  # NCKHW
        log_scales = torch.clamp(
            l[:, 2, ...], min=configs._LOG_SCALES_MIN)  # NCKHW, is >= -7
        x = x.reshape(N, C, 1, H, W)

        if self.use_coeffs:
            # Coefficients only supported for multiples of 3,
            # see note where we define
            # _NUM_PARAMS_RGB NCKHW, basically coeffs_g_r, coeffs_b_r, coeffs_b_g
            assert C == 3, C
            # Each NCKHW
            coeffs = self._nonshared_coeffs_act(l[:, 3, ...])
            # each NKHW
            coeffs_g_r = coeffs[:, 0, ...]
            coeffs_b_r = coeffs[:, 1, ...]
            coeffs_b_g = coeffs[:, 2, ...]
            # NCKHW
            means = torch.stack(
                (means[:, 0, ...],
                 means[:, 1, ...] + coeffs_g_r * x[:, 0, ...],
                 means[:, 2, ...] + coeffs_b_r * x[:, 0, ...]
                                  + coeffs_b_g * x[:, 1, ...]),
                dim=1)

        means = torch.clamp(means, min=self.x_min, max=self.x_max)
        assert means.shape == (N, C, K, H, W), (means.shape, (N, C, K, H, W))
        return x, logit_probs, means, log_scales, K


class LogisticMixtureProbability(NamedTuple):
    name: str
    pixel_index: int
    probs: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


class ResBlock(nn.Module):

    def __init__(self,
                 n_feats: int,
                 kernel_size: int,
                 act: str = "leaky_relu",
                 atrous: int = 1,
                 bn: bool = False) -> None:
        """ param n_feats: Channel size.
            param kernel_size: kernel size.
            param act: string of activation to use.
            param atrous: controls amount of dilation to use in final conv.
            param bn: Turns on batch norm. 
        """
        super().__init__()

        m: List[nn.Module] = []
        _repr = []
        for i in range(2):
            atrous_rate = 1 if i == 0 else atrous
            conv_filter = conv(
                n_feats, n_feats, kernel_size, rate=atrous_rate, bias=True)
            m.append(conv_filter)
            _repr.append(f"Conv({n_feats}x{kernel_size}" +
                         (f";A*{atrous_rate})" if atrous_rate != 1 else "") +
                         ")")

            if bn:
                m.append(nn.BatchNorm2d(n_feats))
                _repr.append(f"BN({n_feats})")

            if i == 0:
                m.append(get_act(act))
                _repr.append("Act")
        self.body = nn.Sequential(*m)

        self._repr = "/".join(_repr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        res = self.body(x)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self,
                 scale: int,
                 n_feats: int,
                 bn: bool = False,
                 act: str = "none",
                 bias: bool = True) -> None:
        m: List[nn.Module] = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                m.append(get_act(act))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            m.append(get_act(act))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSRDec(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 resblocks: int = 8,
                 kernel_size: int = 3,
                 tail: str = "none",
                 channel_attention: bool = False) -> None:
        super().__init__()
        self.head = conv(in_ch, out_ch, 1)
        m_body: List[nn.Module] = [
            ResBlock(out_ch, kernel_size) for _ in range(resblocks)]

        m_body.append(conv(out_ch, out_ch, kernel_size))
        self.body = nn.Sequential(*m_body)

        self.tail: nn.Module
        if tail == "conv":
            self.tail = conv(out_ch, out_ch, 1)
        elif tail == "none":
            self.tail = nn.Identity()  # type: ignore
        elif tail == "upsample":
            self.tail = Upsampler(scale=2, n_feats=out_ch)
        else:
            raise NotImplementedError(f"{tail} is not implemented.")

    def forward(self,  # type: ignore
                x: torch.Tensor,
                features_to_fuse: torch.Tensor = 0.,  # type: ignore
                ) -> torch.Tensor:
        """
        :param x: N C H W
        :return: N C" H W
        """
        x = self.head(x)
        x = x + features_to_fuse
        x = self.body(x) + x
        x = self.tail(x)
        return x

class Bits:
    """
    Tracks bpsps from different parts of the pipeline for one forward pass.
    """

    def __init__(self) -> None:
        self.key_to_bits: DefaultDict[
            str, torch.Tensor] = defaultdict(float) 
        self.key_to_nll: DefaultDict[
            str, torch.Tensor] = defaultdict(float)
        self.key_to_entropy: DefaultDict[
            str, torch.Tensor] = defaultdict(float)
        self.key_to_sizes: DefaultDict[str, int] = defaultdict(int)

    def add_with_size(
            self, key: str, nll_sum: torch.Tensor, size: int,
    ) -> None:
        if configs.log_likelihood:
            assert key not in self.key_to_bits, f"{key} already exists"
            
            self.key_to_bits[key] = nll_sum.detach().cpu() / np.log(2)
            self.key_to_sizes[key] = size

    def add(self, key: str, nll: torch.Tensor) -> None:
        
        self.add_with_size(
            key, nll.sum(), np.prod(nll.size()))

    def add_lm(
            self, y_i: torch.Tensor,
            lm_probs: LogisticMixtureProbability,
            loss_fn: DiscretizedMixLogisticLoss) -> None:
        assert lm_probs.probs.shape[-2:] == y_i.shape[-2:], (
            lm_probs.probs.shape, y_i.shape)
        if configs.log_likelihood:
            nll, entropy = loss_fn(y_i, lm_probs.probs)
            self.add_full(lm_probs.name, nll, entropy)

    def add_full(
            self, key: str, nll: torch.Tensor, entropy: torch.Tensor
    ) -> None:
        if configs.log_likelihood:
            assert key not in self.key_to_bits, f"{key} already exists"
            self.key_to_nll[key] = nll.detach().cpu()
            self.key_to_entropy[key] = entropy.detach().cpu()

    def get_bits(self, key: str) -> torch.Tensor:
        return self.key_to_bits[key]

    def get_size(self, key: str) -> int:
        return self.key_to_sizes[key]

    def get_keys(self) -> KeysView:
        return self.key_to_nll.keys()

    def get_nll(self, key: str) -> torch.Tensor:
        return self.key_to_nll[key] 
    
    def get_entropy(self, key: str) -> torch.Tensor:
        return self.key_to_entropy[key]

    
    def get_scaled_bpsp(self, key: str) -> torch.Tensor:
        return self.key_to_bits[key]/self.key_to_sizes[key]

    def update(self, other: "Bits") -> "Bits":
        # Used by Compressor to aggregate bits from decoder.
        assert len(self.get_keys() & other.get_keys()) == 0, \
            f"{self.get_keys()} and {other.get_keys()} intersect."
        self.key_to_bits.update(other.key_to_bits)
        self.key_to_nll.update(other.key_to_nll)
        self.key_to_entropy.update(other.key_to_entropy)
        self.key_to_sizes.update(other.key_to_sizes)
        return self

    def add_bits(self, other: "Bits") -> "Bits":
        keys = other.get_keys()
        assert keys == self.get_keys() or len(self.get_keys()) == 0, (
            f"{self.get_keys()} != {keys}")

        for key in keys:
            self.key_to_bits[key] += other.get_bits(key)
            self.key_to_nll[key] += other.get_nll(key).detach().cpu()
            self.key_to_entropy[key] += other.get_entropy(key).detach().cpu()
            self.key_to_sizes[key] += other.get_size(key)
        return self

class StrongPixDecoder(nn.Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        # Input: N 3 H W
        # Output: N C H W
        self.loss_fn = DiscretizedMixLogisticLoss(rgb_scale=True)
        self.scale = scale
        self.rgb_decs = nn.ModuleList([
            EDSRDec(
                3*i, configs.n_feats,
                resblocks=configs.resblocks, tail="conv")
            for i in range(1, 4)
        ])
        self.mix_logits_prob_clf = nn.ModuleList([
            AtrousProbabilityClassifier(
                configs.n_feats, C=3, K=configs.K,
                num_params=self.loss_fn._num_params)
            for _ in range(1, 4)
        ])
        self.feat_convs = nn.ModuleList([
            conv(configs.n_feats, configs.n_feats, 3) for _ in range(1, 4)
        ])
        assert (len(self.rgb_decs) == len(self.mix_logits_prob_clf) ==
                len(self.feat_convs)), (
                    f"{len(self.rgb_decs)}, "
                    f"{len(self.mix_logits_prob_clf)}, {len(self.feat_convs)}"
        )

    def forward_probs(
            self,
            x: torch.Tensor,
            ctx: torch.Tensor,
    ) -> Generator[LogisticMixtureProbability, torch.Tensor,
                   Tuple[torch.Tensor, torch.Tensor]]:
        # mode is used to key tensorboard loggings
        mode = "train" if self.training else "eval"
        # x: N 3 H W, [0, 255]
        # pix_sum: N 3 H W, [0, 1020]
        pix_sum = x * 4
        xy_normalized = x / 127.5 - 1
        y_i = torch.tensor([], device=x.device)
        z: torch.Tensor = 0.  # type: ignore

        for i, (rgb_dec, clf, feat_conv) in enumerate(
                zip(self.rgb_decs,  # type: ignore
                    self.mix_logits_prob_clf, self.feat_convs)):
            xy_normalized = torch.cat((xy_normalized, y_i / 127.5 - 1), dim=1)
            z = rgb_dec(xy_normalized, ctx)
            ctx = feat_conv(z)

            probs = clf(z)
            lower = torch.max(
                pix_sum - (3 - i) * 255, torch.tensor(0., device=x.device))
            upper = torch.min(
                pix_sum, torch.tensor(255., device=x.device))

            y_i = yield LogisticMixtureProbability(
                f"{mode}/{self.scale}_{i}", i, probs, lower, upper)
            y_i = pad(y_i, x.shape[-2], x.shape[-1])
            pix_sum -= y_i

        # Last pixel in 2x2 grid should be <= 255 and >= 0
        return pix_sum, ctx

    def forward(self,  # type: ignore
                x: torch.Tensor,
                y: torch.Tensor,
                ctx: torch.Tensor,
                ) -> Tuple[Bits, torch.Tensor]:
        bits = Bits()

        # Check y are filled with integers.
        # y.long().float() == y
        if __debug__:
            not_int = y.long().float() != y
            assert not torch.any(not_int), y[not_int]

        _, _, x_h, x_w = x.size()
        if not isinstance(ctx, float):
            ctx = ctx[..., :x_h, :x_w]

        y_slices = group_2x2(y)
        gen = self.forward_probs(x, ctx)
        
        try:
            for i, y_slice in enumerate(y_slices):
                if i == 0:
                    lm_probs = next(gen)
                else:
                    lm_probs = gen.send(y_slices[i-1])
                _, _, h, w = y_slice.size()
                lm_probs = LogisticMixtureProbability(
                    name=lm_probs.name,
                    pixel_index=lm_probs.pixel_index,
                    probs=lm_probs.probs[..., :h, :w],
                    lower=lm_probs.lower[..., :h, :w],
                    upper=lm_probs.upper[..., :h, :w])
                bits.add_lm(y_slice, lm_probs, self.loss_fn)
        except StopIteration as e:
            last_pixels, ctx = e.value
            last_slice = y_slices[-1]
            _, _, last_h, last_w = last_slice.size()
            last_pixels = last_pixels[..., : last_h, : last_w]
            assert torch.all(last_pixels == last_slice), (
                last_pixels[last_pixels != last_slice],
                last_slice[last_pixels != last_slice])

        return bits, ctx


class Compressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        assert configs.scale >= 0, configs.scale

        # self.loss_fn = DiscretizedMixLogisticLoss(rgb_scale=True)
        self.ctx_upsamplers = nn.ModuleList([
            nn.Identity(),  # type: ignore
            *[Upsampler(scale=2, n_feats=configs.n_feats)
              for _ in range(configs.scale-1)]
        ] if configs.scale > 0 else [])
        self.decs = nn.ModuleList([
            StrongPixDecoder(i) for i in range(configs.scale)
        ])
        assert len(self.ctx_upsamplers) == len(self.decs), \
            f"{len(self.ctx_upsamplers)}, {len(self.decs)}"
        self.nets = nn.ModuleList([
            self.ctx_upsamplers, self.decs,
        ])

    def forward(self, x: torch.Tensor) -> Bits:
        
        downsampled = average_downsamples(x)
        bits = Bits()
        ctx = 0.
        
        for dec, ctx_upsampler, x, y, in zip(
                self.decs, self.ctx_upsamplers,
                downsampled[::-1], downsampled[-2::-1]):
            ctx = ctx_upsampler(ctx)
            dec_bits, ctx = dec(x, tensor_round(y), ctx)                   # tensor_round(y)
            bits.update(dec_bits)
        
        return bits


class ImageFolder(data.Dataset):

    def __init__(self, file_paths: List[str]) -> None:
        self.file_paths = file_paths

    def to_tensor_not_normalized(self, pic: Image) -> torch.Tensor:
    
        if isinstance(pic, np.ndarray):
            return torch.from_numpy(pic.transpose((2, 0, 1)))

        # Handle different PIL image modes
        mode_to_dtype = {
            'I': np.int32,
            'I;16': np.int16,
            'F': np.float32,
            '1': np.uint8,
            'RGB': np.uint8
        }
        
        if pic.mode in mode_to_dtype:
            img = torch.from_numpy(np.array(pic, mode_to_dtype[pic.mode], copy=True))
            if pic.mode == '1':
                img *= 255

        # Determine number of channels
        nchannel = 3 if pic.mode in ['YCbCr', 'RGB'] else (1 if pic.mode == 'I;16' else len(pic.mode))
        
        # Convert to CHW format
        img = img.view(pic.size[1], pic.size[0], nchannel).transpose(0, 1).transpose(0, 2).contiguous()
        # print(img.size())
        return img.float()

    def load(self, file_path: str) -> torch.Tensor:
        img = Image.open(file_path)
        return self.to_tensor_not_normalized(img)

    def __getitem__(self, idx: int) -> torch.Tensor: 
        file_path = self.file_paths[idx]
        img = self.load(file_path)
        return img

    def __len__(self) -> int:
        return len(self.file_paths)


def to_sym(x, x_min, x_max, L):
    sym_range = x_max - x_min
    bin_size = sym_range / (L-1)
    return x.clamp(x_min, x_max).sub(x_min).div(bin_size).round()


def non_shared_get_Kp(K, C, num_params):
    """ Get Kp=number of channels to predict. 
        See note where we define _NUM_PARAMS_RGB above """
    return num_params * C * K


def non_shared_get_K(Kp: int, C: int, num_params: int) -> int:
    """ Inverse of non_shared_get_Kp, get back K=number of mixtures """
    return Kp // (num_params * C)


def get_act(act: str, n_feats: int = 0) -> nn.Module:
    """ param act: Name of activation used.
        n_feats: channel size.
        returns the respective activation module, or raise
            NotImplementedError if act is not implememted.
    """
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "prelu":
        return nn.PReLU(n_feats)
    elif act == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif act == "none":
        return nn.Identity()  # type: ignore
    raise NotImplementedError(f"{act} is not implemented")


def conv(in_channels: int,
         out_channels: int,
         kernel_size: int,
         bias: bool = True,
         rate: int = 1,
         stride: int = 1) -> nn.Conv2d:
    padding = kernel_size // 2 if rate == 1 else rate
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, dilation=rate,
        padding=padding, bias=bias)


def tensor_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x - 0.001)


def pad(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    _, _, xH, xW = x.size()
    padding = [0, W - xW, 0, H - xH]
    return F.pad(x, padding, mode="replicate")


def pad_to_even(x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.size()
    pad_right = w % 2 == 1
    pad_bottom = h % 2 == 1
    padding = [0, 1 if pad_right else 0, 0, 1 if pad_bottom else 0]
    x = F.pad(x, padding, mode="replicate")
    return x


def average_downsamples(x: torch.Tensor) -> List[torch.Tensor]:
    downsampled = []
    for _ in range(configs.scale):
        downsampled.append(x.detach())
        x = F.avg_pool2d(pad_to_even(tensor_round(x)), 2)
    downsampled.append(x.detach())
    return downsampled


def group_2x2(
        x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Group 2x2 patches of x on its own channel
        param x: N C H W
        returns: Tuple[N 4 C H/2 W/2]
    """
    _, _, h, w = x.size()
    # assert h % 2 == 0, f"{x.shape} does not satisfy h % 2 == 0"
    # assert w % 2 == 0, f"{x.shape} does not satisfy w % 2 == 0"
    x_even_height = x[:, :, 0:h:2, :]
    x_odd_height = x[:, :, 1:h:2, :]
    return (
        x_even_height[:, :, :, 0:w:2],
        x_even_height[:, :, :, 1:w:2],
        x_odd_height[:, :, :, 0:w:2],
        x_odd_height[:, :, :, 1:w:2])
