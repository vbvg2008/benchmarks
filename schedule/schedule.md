
### finished


### to do list:

* refactor all datasets
    * imbalanced batching behavior (both probability and deterministic)
    * dataset padding (like object detection)

* pggan

* increase effective batch size functionality

* compression implementation

* mixed precision



# some back log:
 Update = reduce + update
 gradient_Op = reduce + gradient

* TVM? Ray?
for mode in modes:
    0. get pipeline all outputs
    1. get network all outputs
    2. get network effective inputs (using certain algorithm: two sets are involved)
    3. assert T_inputs is member of  U(P_outputs, N_outputs)
    4. get network effective output:  Inter(T_inputs, N_outputs)

* Introduce FE to Academia in Europe


# issues:
* How to do patching ?
* need to improve the tensorflow model schedule GPU memory usage
* understand tensorflow gpu usage
* tensorflow make distributed value needs to change
* how to pad tensor in dataloder
* how to select certain distribution of labels within batch in data loader




## ideas
pipeline.transform()
network.show_results(pipeline=pipeline, epoch=0, mode="train")
network.transform()


Still needed:
* more API polishing
* update all tutorials
* update all apphubs
* doc strings

How to do inferencing?


