import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from external.weight_normalization import weight_normalization as WN


class IAFLayer(chainer.Chain):
    """
    """
    def __call__(self, x):
        """
        """
        pass

    def forward_up(self, x):
        """
        """
        h = F.elu(x)
        h = self.up1(h)
        sections = [self.z_dim, self.z_dim*2, self.z_dim*2+self.h_dim]
        self.qz_mean, self.qz_logv, self.up_context, h = \
            F.split_axis(h, sections, axis=1)

        h = F.elu(h)
        h = self.up2(h)

        if self.downsample:
            output_shape = h.shape[2:]
            x = F.resize_images(x, output_shape)

        return x + 0.1 * h

    def forward_down(self, x, sample=False):
        """
        """
        h = F.elu(x)
        h = self.down1(h)
        sections = [self.z_dim, self.z_dim*2, self.z_dim*3,
                    self.z_dim*4, self.z_dim*4+self.h_dim]
        pz_mean, pz_logv, rz_mean, rz_logv, down_context, h_det = \
            F.split_axis(h, sections, axis=1)

        prior = F.gaussian(pz_mean, 2 * pz_logv)
        logps = self.gaussian_diag_logps(pz_mean, 2*pz_logv, prior)

        if sample:
            z = prior
            context = 0
            logqs = chainer.Variable(
                self.xp.zeros(logps.shape, dtype="float32"), name="logqs")
        else:
            post_mean = rz_mean + self.qz_mean
            post_logv = 2 * (rz_logv + self.qz_logv)
            posterior = F.gaussian(post_mean, post_logv)
            context = self.up_context + down_context
            logqs = self.gaussian_diag_logps(post_mean, post_logv, posterior)

            z = posterior

        # autoregressive nn
        h = self.ar1(z)
        h = h + context
        h = self.ar2(h)
        sections = [self.z_dim]
        arw_mean, arw_logv = F.split_axis(h, sections, axis=1)
        # arw_mean, arw_logv = h[0] * 0.1, h[1] * 0.1  # ??
        z = (z - 0.1*arw_mean) / F.exp(F.clip(0.1*arw_logv, -100., 100.))
        logqs += arw_logv

        kl_cost = logqs - logps
        kl_cost, kl_obj = self.kl_sum(kl_cost)

        z = F.concat([z, h_det])
        z = F.elu(z)
        z = self.down2(z)
        if self.downsample:
            output_shape = z.shape[2:]
            x = F.resize_images(x, output_shape)

        z = x + 0.1 * z
        return z, kl_obj, kl_cost

    def kl_sum(self, kl_cost):
        """
        """
        raise NotImplemented

    @staticmethod
    def gaussian_diag_logps(mean, logvar, sample=None):
        """
        """
        xp = cuda.get_array_module(mean)
        if sample is None:
            noise = xp.random.standard_normal(mean.shape)
            sample = mean + xp.exp(F.clip(0.5 * logvar, -100., 100.)) * noise

        output = -0.5 * (xp.log(2.0*xp.pi) +
                         logvar + F.square(sample - mean) /
                         F.exp(F.clip(logvar, -100., 100.)))

        return output


class ConvIAFLayer(IAFLayer):
    """
    """
    def __init__(self, in_dim, z_dim, h_dim, ksize, pad,
                 downsample=False):
        """
        """
        super(ConvIAFLayer, self).__init__()
        self.downsample = downsample
        self.z_dim = z_dim
        self.h_dim = h_dim
        up1_dim = 2*z_dim + 2*h_dim
        down1_dim = 4*z_dim + 2*h_dim
        stride = 2 if downsample else 1

        with self.init_scope():
            self.up1 = WN.convert_with_weight_normalization(
                L.Convolution2D, in_dim, up1_dim,
                ksize=3, stride=stride, pad=pad)
            self.up2 = WN.convert_with_weight_normalization(
                L.Convolution2D, h_dim, h_dim,
                ksize=3, stride=1, pad=pad)
            self.down1 = WN.convert_with_weight_normalization(
                L.Convolution2D, in_dim, down1_dim,
                ksize=3, stride=1, pad=pad)
            if self.downsample:
                self.down2 = WN.convert_with_weight_normalization(
                    L.Deconvolution2D, z_dim+h_dim, h_dim,
                    ksize=4, stride=stride, pad=pad)
            else:
                self.down2 = WN.convert_with_weight_normalization(
                    L.Convolution2D, z_dim+h_dim, h_dim,
                    ksize=3, stride=1, pad=pad)
            self.ar1 = WN.convert_with_weight_normalization(
                L.Convolution2D, z_dim, h_dim,
                ksize=3, stride=1, pad=pad)
            self.ar2 = WN.convert_with_weight_normalization(
                L.Convolution2D, h_dim, z_dim*2,
                ksize=3, stride=1, pad=pad)

        self.qz_mean = None
        self.qz_logv = None
        self.up_context = None
        self.kl_min = 0.001

    def kl_sum(self, kl_cost):
        """
        """
        if self.kl_min > 0:
            batch_size = kl_cost.shape[0]
            kl_cost_23 = F.sum(kl_cost, axis=(2, 3))
            kl_ave = F.mean(kl_cost_23, axis=(0), keepdims=True)
            kl_min = self.xp.array([self.kl_min], dtype="float32")
            kl_ave = F.maximum(kl_ave, F.tile(kl_min, kl_ave.shape))
            kl_ave = F.tile(kl_ave, (batch_size, 1))
            kl_obj = F.sum(kl_ave, axis=1)
        else:
            kl_obj = F.sum(kl_cost, axis=(1, 2, 3))

        kl_cost = F.sum(kl_cost, axis=(1, 2, 3))
        return kl_cost, kl_obj


class LinearIAFLayer(IAFLayer):
    """
    """
    def __init__(self, in_dim, z_dim, h_dim):
        """
        """
        super(LinearIAFLayer, self).__init__()
        self.downsample = False
        self.z_dim = z_dim
        self.h_dim = h_dim
        up1_dim = 2*z_dim + 2*h_dim
        down1_dim = 4*z_dim + 2*h_dim

        with self.init_scope():
            self.up1 = WN.convert_with_weight_normalization(
                L.Linear, in_dim, up1_dim
            )
            self.up2 = WN.convert_with_weight_normalization(
                L.Linear, h_dim, h_dim
            )
            self.down1 = WN.convert_with_weight_normalization(
                L.Linear, in_dim, down1_dim
            )
            self.down2 = WN.convert_with_weight_normalization(
                L.Linear, z_dim+h_dim, h_dim
            )
            self.ar1 = WN.convert_with_weight_normalization(
                L.Linear, z_dim, h_dim
            )
            self.ar2 = WN.convert_with_weight_normalization(
                L.Linear, h_dim, z_dim*2
            )

        self.qz_mean = None
        self.qz_logv = None
        self.up_context = None
        self.kl_min = 0.001

    def kl_sum(self, kl_cost):
        """
        """
        if self.kl_min > 0:
            batch_size = kl_cost.shape[0]
            kl_cost_23 = F.sum(kl_cost, axis=(1))
            kl_ave = F.mean(kl_cost_23, axis=(0), keepdims=True)
            kl_min = self.xp.array([self.kl_min], dtype="float32")
            kl_ave = F.maximum(kl_ave, F.tile(kl_min, kl_ave.shape))
            kl_ave = F.tile(kl_ave, (batch_size, 1))
            kl_obj = F.sum(kl_ave, axis=1)
        else:
            kl_obj = F.sum(kl_cost, axis=(1))

        kl_cost = F.sum(kl_cost, axis=(1))
        return kl_cost, kl_obj
