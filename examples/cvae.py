import chainer
import chainer.links as L
import chainer.functions as F
from links.connection.iaf import IAFLayer
from chainer import report
from external.weight_normalization import weight_normalization as WN


class IAFCVAE(chainer.Chain):
    """
    """
    def __init__(self,
                 in_channel=1,
                 h_channel=3,
                 depth=1,
                 n_iaf_block=1,
                 iaf_params=dict()):
        """
        """
        super(IAFCVAE, self).__init__()
        self.depth = depth
        self.n_iaf_block = n_iaf_block

        with self.init_scope():
            # encoder
            self.conv1 = WN.convert_with_weight_normalization(
                L.Convolution2D, in_channel, h_channel,
                ksize=3, stride=2, pad=1
            )
            # self.conv1 = L.Convolution2D(in_channel,
            #                              h_channel,
            #                              ksize=3,
            #                              stride=2,
            #                              pad=1)
            # IAF layers
            iaf_layers = list()
            for i in range(depth):
                for j in range(n_iaf_block):
                    downsample = (i > 0) and (j == 0)
                    iaf_layers.append(
                        IAFLayer(downsample=downsample, **iaf_params)
                    )
            self.iaf_layers = chainer.ChainList(*iaf_layers)
            # decoder
            self.deconv1 = WN.convert_with_weight_normalization(
                L.Deconvolution2D, h_channel, in_channel,
                ksize=4, stride=2, pad=1)
            # self.deconv1 = L.Deconvolution2D(h_channel,
            #                                  in_channel,
            #                                  ksize=4,
            #                                  stride=2,
            #                                  pad=1)
            self.h_top = chainer.Variable(
                self.xp.zeros(h_channel, dtype="float32"), name="h_top")

    def __call__(self, x):
        """
        """
        x = x - 0.5
        batch_size = x.shape[0]
        image_size = x.shape[-1]
        data_size = batch_size

        # encode
        h = self.conv1(x)
        # IAF up
        for iaf_layer in self.iaf_layers:
            h = iaf_layer.forward_up(h)

        # IAF down
        h_top = F.reshape(self.h_top, [1, -1, 1, 1])
        wh_size = int(image_size/(2**self.depth))
        h = F.tile(h_top, (data_size, 1, wh_size, wh_size))

        kl_cost = kl_obj = 0.0
        for iaf_layer in self.iaf_layers[::-1]:
            h, cur_obj, cur_cost = iaf_layer.forward_down(h)
            kl_obj += cur_obj
            kl_cost += cur_cost

        x_hat = F.elu(h)
        x_hat = self.deconv1(x_hat)
        x_hat = F.clip(x_hat, -0.5 + 1/512., 0.5 - 1/512.)

        # discretize_logits nisuru
        log_pxz = -F.bernoulli_nll(x_hat, x, reduce="no")
        log_pxz = F.sum(log_pxz, axis=(1, 2, 3))
        obj = F.sum(kl_obj - log_pxz)/batch_size
        recon_loss = F.sum(log_pxz)/batch_size

        return x_hat, obj, recon_loss

    def loss(self, x):
        """
        """
        _, loss_value, recon_loss = self(x)
        report({"loss": loss_value, "recon_loss": recon_loss}, self)
        return loss_value
