import tempfile

import fastestimator as fe
from fastestimator.architecture.pytorch import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy

import sls


class DummpyUpdate(UpdateOp):
    def forward(self, data, state):
        pass


class SGDLinesSearch(fe.op.tensorop.TensorOp):
    def __init__(self, model, opt, loss_op, inputs, outputs, mode="train"):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.model = model
        self.opt = opt
        self.loss_op = loss_op

    def forward(self, data, state):
        x, y = data
        closure = lambda: self.loss_op.forward((self.model(x), y), state=state)
        self.opt.zero_grad()
        loss = self.opt.step(closure=closure)
        return loss


def get_estimator(epochs=2, batch_size=32, save_dir=tempfile.mkdtemp()):
    # step 1
    train_data, eval_data = mnist.load_data()

    pipeline = fe.Pipeline(train_data=train_data,
                           eval_data=eval_data,
                           batch_size=batch_size,
                           ops=[ExpandDims(inputs="x", outputs="x", axis=0), Minmax(inputs="x", outputs="x")])
    # step 2
    model = fe.build(model_fn=LeNet, optimizer_fn="adam")
    opt = sls.Sls(model.parameters())
    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        SGDLinesSearch(model=model,
                       opt=opt,
                       loss_op=CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
                       inputs=("x", "y"),
                       outputs="ce"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce", mode="eval"),
        DummpyUpdate(model=model, loss_name="ce")
    ])
    # step 3
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max")
    ]
    estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    return estimator
