

#=================Case 1==========================================================
#tf.data.Dataset users
pipeline = fe.Pipeline(train_data=ds_train, eval_data=ds_eval)
network = fe.Network(...)
estimator = fe.Estimator(pipeline=pipeline, ...)


#Torch data loader users
pipeline = fe.Pipeline(train_data=dl_train, eval_data=dl_eval)
network = fe.Network(...)
estimator = fe.Estimator(pipeline=pipeline, ...)


#FE users
pipeline = fe.Pipeline(train_data=fe_ds_train, eval_data=fe_ds_eval, ops=[], ...)
network = fe.Network(...)
estimator = fe.Estimator(pipeline=pipeline)

#--------
"""
Strength: 
1. consistent API usage flow: Pipeline -> Network -> Estimator
2. torch loader users can leveraging pipeline.benchmark(), pipeline.show_batch()
3. transparent: estimator.pipeline is actually pipeline instance

Weakness:
1. have to know certain arguments are for fe pipeline only (common among APIs, eg, torch data loader, Inception model)
"""


#=================Case 2=========================================================
#tf.data.Dataset users
network = fe.Network(...)
estimator = fe.Estimator(pipeline={"train": ds_train, "eval": ds_eval}, ...)

#Torch data loader users
network = fe.Network(...)
estimatpr = fe.Estimatpr(pipeline={"train": dl_train, "eval": dl_eval}, ...)


#FE users
pipeline = fe.Pipeline(train_data=fe_ds_train, eval_data=fe_ds_eval, ops=[], ...)
network = fe.Network(...)
estimator = fe.Estimator(pipeline=pipeline)


#--------
"""
Strength: 
1. dedicated API arguments for fe.Pipeline

Weakness:
1. inconsistent API flow (no longer pipeline -> Network -> Estimator)
2. can't leverage pipeline.benchmark(), pipeline.show_batch()
3. not transparent
"""



#========Common ground========

#Inheritenace tree:
#case 1
BasePipeline
    |-fe.Pipeline
    |-TensorFlowPipeline
    |-TorchPipeline


#case 2

fe.Pipeline 


BasePipeline
    |-FastEstimatorPipeline
    |-TensorFlowPipeline
    |-TorchPipeline



#=============comments:
#1. infer batch size may not work (e.g. patching, batch size is 2, but create 16 patches from each example. the batch size should still be 2)

#2. need functionality to combine batch dimension and patch dimension


class Pipeline:
    def __init__(self,):
        if 
        self.pipeline = TensorFlowPipeline()


    def get_batch_size():
        self.pipeline.get_batch_size()

    
