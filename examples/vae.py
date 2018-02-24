import chainer
import chainer.links as L
import chainer.functions as F
from links.connection.iaf import LinearIAFLayer
from chainer import report
from external.weight_normalization import weight_normalization as WN


class IAFVAE(chainer.Chain):
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
        super(IAFVAE, self).__init__()
        self.depth = depth
        self.n_iaf_block = n_iaf_block

        with self.init_scope():
            # encoder
            self.encoder1 = WN.convert_with_weight_normalization(
                L.Linear, in_channel, in_channel
            )
            self.encoder2 = WN.convert_with_weight_normalization(
                L.Linear, in_channel, h_channel
            )
            # IAF layers
            iaf_layers = list()
            for i in range(depth):
                for j in range(n_iaf_block):
                    iaf_layers.append(
                        LinearIAFLayer(**iaf_params)
                    )
            self.iaf_layers = chainer.ChainList(*iaf_layers)
            # decoder
            self.decoder1 = WN.convert_with_weight_normalization(
                L.Linear, h_channel, in_channel
            )
            self.decoder2 = WN.convert_with_weight_normalization(
                L.Linear, in_channel, in_channel
            )
            self.h_top = chainer.Variable(
                self.xp.zeros(h_channel, dtype="float32"), name="h_top")

    def __call__(self, x):
        """
        """
        x = x - 0.5
        batch_size = x.shape[0]

        # encode
        h = self.encoder1(x)
        h = self.encoder2(h)
        # IAF up
        for iaf_layer in self.iaf_layers:
            h = iaf_layer.forward_up(h)

        # IAF down
        h_top = F.reshape(self.h_top, [1, -1])
        z = F.tile(h_top, (batch_size, 1))

        kl_cost = kl_obj = 0.0
        for iaf_layer in self.iaf_layers[::-1]:
            z, cur_obj, cur_cost = iaf_layer.forward_down(z)
            kl_obj += cur_obj
            kl_cost += cur_cost

        # decoder
        x_hat = F.elu(z)
        x_hat = self.decoder1(x_hat)
        x_hat = self.decoder2(x_hat)
        x_hat = F.clip(x_hat, -0.5 + 1/512., 0.5 - 1/512.)

        # discretize_logits nisuru
        log_pxz = -F.bernoulli_nll(x_hat, x, reduce="no")
        log_pxz = F.sum(log_pxz, axis=(1))
        obj = F.sum(kl_obj - log_pxz)/batch_size
        recon_loss = -F.sum(log_pxz)/batch_size

        return x_hat, obj, recon_loss

    def loss(self, x):
        """
        """
        _, loss_value, recon_loss = self(x)
        report({"loss": loss_value, "recon_loss": recon_loss}, self)
        return loss_value

    def generate(self, h_top, image_size=28, batch_size=1):
        """
        """
        # IAF down
        h_top = F.reshape(h_top, [1, -1, 1, 1])
        wh_size = int(image_size/(2**self.depth))
        z = F.tile(h_top, (batch_size, 1, wh_size, wh_size))

        for iaf_layer in self.iaf_layers[::-1]:
            z, _, _ = iaf_layer.forward_down(z, sample=True)

        # decoder
        x_hat = F.elu(z)
        x_hat = self.decoder1(x_hat)
        x_hat = F.clip(x_hat, -0.5 + 1/512., 0.5 - 1/512.)

        return x_hat
