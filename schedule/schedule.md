
### to do list:

* take care of tensor ranks when using generator
* 



* add multi-gpu, do benchmark

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

# issues:
* How to do patching in pytorch?
* tensorflow dataset sometimes stuck at evaluation
* consider dropping the typing in development
* should we still consider the concatenation ops rule?
* need to improve the tensorflow model schedule GPU memory usage

## ideas
pipeline.transform()


network.show_results(pipeline=pipeline, epoch=0, mode="train")
network.transform()


Still needed:
* more API polishing
* multi-gpu
* update all tutorials
* update all apphubs
* doc strings


