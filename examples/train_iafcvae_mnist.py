import chainer
from chainer import datasets, serializers, iterators, optimizers, training
from chainer.training import extensions
from cvae import IAFCVAE


if __name__ == "__main__":
    result_dir = "result/cae-mnist"
    train, test = datasets.get_mnist(withlabel=False, ndim=3)
    train_iter = iterators.SerialIterator(train,
                                          batch_size=128,
                                          shuffle=True)
    test_iter = iterators.SerialIterator(test,
                                         batch_size=64,
                                         shuffle=False,
                                         repeat=False)

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

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    serializers.load_npz(result_dir + "/model_weights.npz", model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    updater = training.StandardUpdater(train_iter,
                                       optimizer,
                                       loss_func=model.loss)
    trainer = training.Trainer(updater,
                               stop_trigger=(10, "epoch"),
                               out=result_dir)
    trainer.extend(extensions.Evaluator(test_iter, model,
                                        eval_func=model.loss))
    trainer.extend(extensions.dump_graph("main/loss"))
    trainer.extend(
        extensions.snapshot(filename="snapshot_{.updater.epoch}.npz"),
        trigger=(5, "epoch")
    )
    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ["main/loss", "main/recon_loss",
                 "validation/main/loss", "validation/main/recon_loss"],
                "epoch", file_name="loss.png"
            )
        )
    trainer.extend(
        extensions.PrintReport(
            ["epoch",
             "main/loss", "main/recon_loss",
             "validation/main/loss", "validation/main/recon_loss",
             "elapsed_time"]
        )
    )
    trainer.extend(extensions.ProgressBar())
    # serializers.load_npz(result_dir + "/snapshot_20.npz", trainer)
    trainer.run()
    serializers.save_npz(result_dir + "/model_weights.npz", model)
