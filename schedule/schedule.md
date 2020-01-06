### To-do list

* new FE pipeline design - MP
	* move pipeline to new codebase
    * Dataset class (csv dataset, folder structure dataset, in-memory dataset, generator dataset)
        len(dataset)
        dataset[0]
    * integrate albumentation with Operator
    * fix pipeline multi-processing

* new app hub example - JP



Eng work
* windows cli (see pip install) 
* nightly built website ?  if needed to update the docstrings or tutorials/examples, automatic serve?
* docstring md format? what do we need?
* Version control in FE documentation page -

## ideas

Pipeline:
	- output filter mechanism, only output what's used by Network and Trace


Network:
	- input filter mechanism, only send network related data to gpu
	- output filter mechanism, only return Trace inputs from gpu to cpu - prediction

Estimator:
	batch (from cpu) prediction(from gpu)





pipeline.benchmark()
it = pipeline.get_iterator(epoch=0, mode="train")
len(it)
it[0]

pipeline.transform()


network.show_results(pipeline=pipeline, epoch=0, mode="train")
network.transform()




## Showcase

1. new version usage (pytorch user, tensorflow user)

2. backend 

3. improvements: memory consumption

	Pipeline:  A(inputs="x", )

4. Trace interface change
