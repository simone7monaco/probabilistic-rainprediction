import torch
import torch.nn as nn
import numpy as np

conv_out_shape = lambda W: int(W - 3 + 2) + 1

class EncBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.):
        super().__init__()
        # add also the dropout layer
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding='same'),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.in_channels = in_c
        self.out_channels = out_c
        
    def forward(self, inputs):
        x = self.net(inputs)
        x = self.dropout(x)
        p = self.pool(x)
        return x, p
    
class DecBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = EncBlock(out_c+out_c, out_c, dropout=dropout)
        self.in_channels = in_c
        self.out_channels = out_c
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x, _ = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout=0., channels=[64, 128, 256, 512, 1024]):
        super().__init__()

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.

        self.dropout = dropout
        channels = [in_features] + channels
        self.encs = nn.ModuleList([
            EncBlock(channels[i], channels[i+1], dropout) for i in range(len(channels)-1)
        ])
        
        # Decoder
        self.decs = nn.ModuleList([
            DecBlock(channels[i], channels[i-1], dropout) for i in range(len(channels)-1, 1, -1)
        ])
        # Output layer
        self.outconv = nn.Conv2d(64, out_features, kernel_size=1)
    
    def eval(self):
        super().eval()
        if self.dropout > 0:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
            

    def encoder(self, p):
        skips = []
        for block in self.encs:
            x, p = block(p)
            skips.append(x)
        return skips
    
    def decoder(self, skips):
        x = skips[-1]
        for block, skip in zip(self.decs, reversed(skips[:-1])):
            x = block(x, skip)
        return x
    
    def forward(self, x):
        skips = self.encoder(x)
        x = self.decoder(skips)
        out = self.outconv(x)
        return out


class VUNet(UNet):
    def __init__(self, in_features: int, out_features: int, dropout=0.):
        super().__init__(in_features, out_features, dropout)
        self.mu = EncBlock(1024, 1024)
        self.logvar = EncBlock(1024, 1024)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        x, skips = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z, skips)
        out = self.outconv(x)
        return out, mu, logvar


class AttentionBlock(nn.Module):
    def __init__(self, input_dim, output_shape, hidden_dim=64):
        super().__init__()
        assert len(output_shape) == 2, "output_shape should have the form L x W of the desired output, excluding channels"
        self.output_shape = output_shape
        self.fc = nn.Linear(input_dim, output_shape[0] * output_shape[1])
        # TODO: add a second layer?
            
    def forward(self, date, x):
        attention_mask = torch.sigmoid(self.fc(date))
        attention_mask = attention_mask.view(-1, 1, self.output_shape[0], self.output_shape[1])
        return x * attention_mask
        
class ExtraUNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, time_features:int=8, image_shape=(96, 128), activation=None, use_batchnorm=False, use_attention=False):
        super().__init__()

        self.activation = activation
        image_shape = np.array(image_shape)
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------

        # Maxpooling in the last block
        self.encoder = nn.ModuleList([
            EncBlock(in_features, 64, use_batchnorm=use_batchnorm), # image shape after pooling: h/2 x w/2
            EncBlock(64, 128, use_batchnorm=use_batchnorm), # h/4 x w/4
            EncBlock(128, 256, use_batchnorm=use_batchnorm), # h/8 x w/8
            EncBlock(256, 512, use_batchnorm=use_batchnorm) # h/16 x w/16
        ])
        
        self.bottleneck = EncBlock(512, 1024, use_batchnorm=use_batchnorm).block
        
        # Decoder
        self.decoder = nn.ModuleList([
            DecBlock(1024, 512, use_batchnorm=use_batchnorm),
            DecBlock(512, 256, use_batchnorm=use_batchnorm),
            DecBlock(256, 128, use_batchnorm=use_batchnorm),
            DecBlock(128, 64, use_batchnorm=use_batchnorm)
        ])

        # Output layer
        self.outconv = nn.Conv2d(64, out_features, kernel_size=1)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention_blocks = nn.ModuleList([
                AttentionBlock(input_dim=time_features, output_shape=image_shape),
                AttentionBlock(input_dim=time_features, output_shape=image_shape//2),
                AttentionBlock(input_dim=time_features, output_shape=image_shape//4),
                AttentionBlock(input_dim=time_features, output_shape=image_shape//8),
            ])
        else:
            self.attention_blocks = [nn.Identity] * 4

    def forward(self, x, date):
        # the "Extra" model has an extra input, the date (a unique number for the whole image), this value is encoded as an attention map and multiplied by the output of all the skip connections            
        # Encoder
        skips = []
        for block, att_block in zip(self.encoder, self.attention_blocks):
            s, x = block(x)
            if self.use_attention:
                s = att_block(date, s)
            skips.append(s)
        x = self.bottleneck(x)

        # Decoder
        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x, skip)

        # Output layer
        out = self.outconv(x)
        return out
