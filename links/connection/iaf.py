import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from external.weight_normalization import weight_normalization as WN


class IAFLayer(chainer.Link):
    """
    """
    def __init__(self, in_dim, z_dim, h_dim, ksize, pad,
                 downsample=False):
        """
        """
        super(IAFLayer, self).__init__()
        self.downsample = downsample
        self.z_dim = z_dim
        self.h_dim = h_dim
        up_conv1_dim = 2*z_dim + 2*h_dim
        down_conv1_dim = 4*z_dim + 2*h_dim
        stride = 2 if downsample else 1

        with self.init_scope():
            self.up_conv1 = WN.convert_with_weight_normalization(
                L.Convolution2D, in_dim, up_conv1_dim,
                ksize=3, stride=stride, pad=pad)
            # self.up_conv1 = L.Convolution2D(in_dim,
            #                                 up_conv1_dim,
            #                                 ksize=3,
            #                                 stride=stride,
            #                                 pad=pad)
            self.up_conv2 = WN.convert_with_weight_normalization(
                L.Convolution2D, h_dim, h_dim,
                ksize=3, stride=1, pad=pad)
            # self.up_conv2 = L.Convolution2D(h_dim,
            #                                 h_dim,
            #                                 ksize=3,
            #                                 stride=1,
            #                                 pad=pad)
            self.down_conv1 = WN.convert_with_weight_normalization(
                L.Convolution2D, in_dim, down_conv1_dim,
                ksize=3, stride=1, pad=pad)
            # self.down_conv1 = L.Convolution2D(in_dim,
            #                                   down_conv1_dim,
            #                                   ksize=3,
            #                                   stride=1,
            #                                   pad=pad)
            if self.downsample:
                self.down_conv2 = WN.convert_with_weight_normalization(
                    L.Deconvolution2D, z_dim+h_dim, h_dim,
                    ksize=4, stride=stride, pad=pad)
                # self.down_conv2 = L.Deconvolution2D(z_dim+h_dim,
                #                                     h_dim,
                #                                     ksize=4,
                #                                     stride=stride,
                #                                     pad=pad)
            else:
                self.down_conv2 = WN.convert_with_weight_normalization(
                    L.Convolution2D, z_dim+h_dim, h_dim,
                    ksize=3, stride=1, pad=pad)
                # self.down_conv2 = L.Convolution2D(z_dim+h_dim,
                #                                   h_dim,
                #                                   ksize=3,
                #                                   stride=1,
                #                                   pad=pad)
            self.ar_conv1 = WN.convert_with_weight_normalization(
                L.Convolution2D, z_dim, h_dim,
                ksize=3, stride=1, pad=pad)
            # self.ar_conv1 = L.Convolution2D(z_dim,
            #                                 h_dim,
            #                                 ksize=3,
            #                                 stride=1,
            #                                 pad=pad)
            self.ar_conv2 = WN.convert_with_weight_normalization(
                L.Convolution2D, h_dim, z_dim*2,
                ksize=3, stride=1, pad=pad)
            # self.ar_conv2 = L.Convolution2D(h_dim,
            #                                 z_dim*2,
            #                                 ksize=3,
            #                                 stride=1,
            #                                 pad=pad)

        self.qz_mean = None
        self.qz_logv = None
        self.up_context = None
        self.kl_min = 0

    def __call__(self, x):
        """
        """
        pass

    def forward_up(self, x):
        """
        """
        h = F.elu(x)
        h = self.up_conv1(h)
        sections = [self.z_dim, self.z_dim*2, self.z_dim*2+self.h_dim]
        self.qz_mean, self.qz_logv, self.up_context, h = \
            F.split_axis(h, sections, axis=1)

        h = F.elu(h)
        h = self.up_conv2(h)

        if self.downsample:
            output_shape = h.shape[2:]
            x = F.resize_images(x, output_shape)

        return x + 0.1 * h

    def forward_down(self, x):
        """
        """
        h = F.elu(x)
        h = self.down_conv1(h)
        sections = [self.z_dim, self.z_dim*2, self.z_dim*3,
                    self.z_dim*4, self.z_dim*4+self.h_dim]
        pz_mean, pz_logv, rz_mean, rz_logv, down_context, h_det = \
            F.split_axis(h, sections, axis=1)

        prior = F.gaussian(pz_mean, 2 * pz_logv)
        post_mean = rz_mean + self.qz_mean
        post_logv = 2 * (rz_logv + self.qz_logv)
        posterior = F.gaussian(post_mean, post_logv)
        context = self.up_context + down_context

        z = posterior
        logqs = self.gaussian_diag_logps(post_mean, post_logv, posterior)
        logps = self.gaussian_diag_logps(pz_mean, 2*pz_logv, prior)

        # autoregressive nn
        h = self.ar_conv1(z)
        h = h + context
        h = self.ar_conv2(h)
        sections = [self.z_dim]
        arw_mean, arw_logv = F.split_axis(h, sections, axis=1)
        # arw_mean, arw_logv = h[0] * 0.1, h[1] * 0.1  # ??
        z = (z - 0.1*arw_mean) / F.exp(0.1*arw_logv)
        logqs += arw_logv

        kl_cost = logqs - logps

        if self.kl_min > 0:
            pass
        else:
            kl_obj = F.sum(kl_cost, axis=(1, 2, 3))

        kl_cost = F.sum(kl_cost, axis=(1, 2, 3))

        h = F.concat([z, h_det])
        h = F.elu(h)
        h = self.down_conv2(h)
        if self.downsample:
            output_shape = h.shape[2:]
            x = F.resize_images(x, output_shape)

        output = x + 0.1 * h
        return output, kl_obj, kl_cost

    @staticmethod
    def gaussian_diag_logps(mean, logvar, sample=None):
        """
        """
        xp = cuda.get_array_module(mean)
        if sample is None:
            noise = xp.random.standard_normal(mean.shape)
            sample = mean + xp.exp(0.5 * logvar) * noise

        output = -0.5 * (xp.log(2.0*xp.pi) +
                         logvar + F.square(sample - mean) / F.exp(logvar))

        return output
