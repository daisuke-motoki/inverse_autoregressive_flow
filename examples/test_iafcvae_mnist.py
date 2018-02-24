from chainer import datasets, serializers
from cvae import IAFCVAE
import matplotlib.pyplot as plt


if __name__ == "__main__":
    result_dir = "result/cae-mnist"
    _, test = datasets.get_mnist(withlabel=False, ndim=3)

    h_channel = 2
    params = dict(
        in_channel=1,
        h_channel=h_channel,
        depth=1,
        n_iaf_block=1,
        iaf_params=dict(
            in_dim=h_channel,
            z_dim=2,
            h_dim=h_channel,
            ksize=3,
            pad=1,
        )
    )
    model = IAFCVAE(**params)
    serializers.load_npz(result_dir + "/model_weights.npz", model)
    import numpy as np
    model.generate(np.array([0.,0.]))
