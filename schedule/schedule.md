
### to do list:

* modify key checking with modes

7. do benchmark on single gpu speed

8. add multi-gpu, do benchmark



## Scheduler ideas:
fe.build
    optimizers: - on network instance   worry about it later

fe.network
    ops - in Networks:        - no the fly


lr schedule



# some back log:
 Update = reduce + update

 gradient_Op = reduce + gradient



* TVM? Ray?

    0. get pipeline all outputs (by creating iterator on the fly) - done
    1. get network all outputs  - done
    2. get network effective inputs (using certain algorithm: two sets are involved)
    3. assert T_inputs is member of  U(P_outputs, N_outputs)
    4. get network effective output:  Inter(T_inputs, N_outputs)

## notebook related

notebook usecase: ?

estimator.fit, then fit again?


# issues:
* How to do patching in pytorch?
* tensorflow dataset sometimes stuck at evaluation
* consider dropping the typing in development


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