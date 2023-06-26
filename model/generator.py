import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .lvcnet import LVCBlock

MAX_WAV_VALUE = 32768.0


class Upsampler(nn.Module):
    
    def __init__(self, hp):
        super(Upsampler, self).__init__()
        self.__hp = hp
        # setting deconvs params
        in_channels = hp.audio.latents_hop_length
        out_channels = hp.audio.n_mel_channels
        middle_channels = in_channels // 2
        stride = 2
        padding = stride // 2
        kernel_size = stride + 2*padding
        bias = False
        ''' finding formule to kernel_size
        The formule to find out_size of deconv1d is:
           out_size = (in_size - 1) * stride + kernel - 2*padding
        We know that: the out_size is equals to 2*in_size
                      padding is equals to stride/2, so [2 / 2 = 1]
                      stride is equals to 2
        lets calculate
                                  2*in_size = (in_size - 1) * stride + kernel - 2*padding
                                  2*in_size = stride*in_size - stride + kernel - 2*padding
        2*in_size - stride*in_size - kernel = -stride - 2*padding
           2*in_size - (2)*in_size - kernel = -stride - 2*padding       [stride = 2]
                                 0 - kernel = -stride - 2*padding
                                     kernel = stride + 2*padding
        '''
        # building deconvs
        self.__deconv1d_lv1 = nn.ConvTranspose1d(in_channels=in_channels,
                                                  out_channels=middle_channels,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  bias=bias)
        
        self.__deconv1d_lv2 = nn.ConvTranspose1d(in_channels=middle_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  bias=bias)
        
        self.__emb_layer = torch.nn.Embedding(8194, self.__hp.audio.latents_hop_length) # generating embedder
    
    def __code_to_emb(self, code: torch) -> torch:
        emb = self.__emb_layer(code.int()) # code(... , 1, T) --> emb (... , T, L)
        # trainnig
        if len(emb.shape) > 3:
            return emb.squeeze(1)
        # validation
        return emb.squeeze(0)
    
    def __upsampling(self, emb: torch) -> torch:
        emb = self.__deconv1d_lv1(emb) # emb (1, LATENTS_HOP_LENGTH, T) -> emb (1, M, T*2)
        emb = self.__deconv1d_lv2(emb) # emb (1, M, T*2) -> emb (1, N_MEL_CHANNELS, (T*2)*2)
        return emb

    def forward(self, code: torch) -> torch:
        emb = self.__code_to_emb(code)
        # prepering embedding to upsample
        emb = emb.transpose(2,1) # emb (1, T, L) --> emb (1, L, T)
        upsampled_emb = self.__upsampling(emb)
        
        return upsampled_emb


class Generator(nn.Module):
    """UnivNet Generator"""
    def __init__(self, hp):
        super(Generator, self).__init__()
        self.mel_channel = hp.audio.n_mel_channels
        self.noise_dim = hp.gen.noise_dim
        self.hop_length = hp.audio.hop_length
        channel_size = hp.gen.channel_size
        kpnet_conv_size = hp.gen.kpnet_conv_size
        self.latents_dim = hp.audio.latents_dim

        # hop length between mel spectrograms and audio
        self.mel_ar_token_ratio = hp.audio.latents_hop_length // hp.audio.hop_length

        self.res_stack = nn.ModuleList()
        hop_length = 1
        for stride in hp.gen.strides:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    hp.audio.n_mel_channels,
                    stride=stride,
                    dilations=hp.gen.dilations,
                    lReLU_slope=hp.gen.lReLU_slope,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size
                )
            )

        self.conv_pre = \
            nn.utils.weight_norm(nn.Conv1d(hp.gen.noise_dim, channel_size, 7, padding=3, padding_mode='reflect'))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(hp.gen.lReLU_slope),
            nn.utils.weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

        self.upsampler = Upsampler(hp)

    def forward(self, c, z):
        '''
        Args:
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length) 
            z (Tensor): the noise sequence (batch, noise_dim, in_length)
        '''
        c_emb = self.upsampler(c)

        z = self.conv_pre(z)                # (B, c_g, L)

        for res_block in self.res_stack:
            res_block.to(z.device)
            z = res_block(z, c_emb)             # (B, c_g, L * s_0 * ... * s_i)

        z = self.conv_post(z)               # (B, 1, L * 256)

        return z

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        print('Removing weight norm...')

        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def inference(self, c, z=None):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8

        #zero = torch.full((1, self.mel_channel, 10), -11.5129).to(c.device)
        zero = torch.full((1, 10), 8193).to(c.device)
        mel = torch.cat((c, zero), dim=-1).unsqueeze(0)
        print(mel.shape)
        if z is None:
#            z = torch.randn(1, self.noise_dim, mel.size(2)).to(mel.device)
            z = torch.randn(1, self.noise_dim, mel.size(2)*self.mel_ar_token_ratio).to(mel.device)

        audio = self.forward(mel, z)
        audio = audio.squeeze() # collapse all dimension except time axis
        audio = audio[:-(self.hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()

        return audio

if __name__ == '__main__':
    hp = OmegaConf.load('../config/default.yaml')
    model = Generator(hp)

    c = torch.randn(3, 100, 10)
    z = torch.randn(3, 64, 10)
    print(c.shape)

    y = model(c, z)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
