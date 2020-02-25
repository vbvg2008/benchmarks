
### to do list:


* create product data type conversion script -> upload data

* pytorch multi-gpu training

* update GAN example

* do normalized conv




* increase effective batch size functionality

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
* How to do patching in pytorch?
* need to improve the tensorflow model schedule GPU memory usage
* understand tensorflow gpu usage
* tensorflow make distributed value needs to change

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

How to do inferencing?


