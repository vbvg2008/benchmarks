### To-do list


* windows cli (see pip install)  - YC

* new FE pipeline design
    * Dataset class (csv dataset, folder structure dataset, in-memory dataset, generator dataset)
        len(dataset)
        dataset[0]

    * integrate albumentation with Operator
    * fix pipeline with multi-processing

* Version control in FE documentation page



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