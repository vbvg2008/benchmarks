## ideas

Pipeline:
	- output filter mechanism, only output what's used by Network and Trace


Network:
	- input filter mechanism, only send network related data to gpu
	- output filter mechanism, only return Trace inputs from gpu to cpu - prediction

Estimator:
	- Collect Trace
	batch (from cpu) prediction(from gpu)



Filtering Logics:

Given: 
* Network all inputs & ouputs (on epoch basis)
* Pipeline all inputs & outputs (on epoch basis)

Get Trace inputs -> Get Network net outputs & inputs -> Get Pipeline net outputs

Get 


pipeline.benchmark()
it = pipeline.get_iterator(epoch=0, mode="train")
len(it)
it[0]

pipeline.transform()


network.show_results(pipeline=pipeline, epoch=0, mode="train")
network.transform()



### to do list:

6. add scheduling functionality

7. do benchmark on single gpu speed

8. add multi-gpu, do benchmark

* make it notebook friendly

## board

notebook usecase: ?

estimator.fit, then fit again?



## Scheduler ideas:

things to schedule: querry dynamincally!

    train_data(not supporting tf.dataset or torch loader scheduling for now)
    val_data
    ops
    batch_size

    optimizers

    ops - in Networks


still needs warm-up

 Update = reduce + update

 gradient_Op = reduce + gradient



* TVM? Ray?



## Today:
* finish new pipeline and scheduling
 


