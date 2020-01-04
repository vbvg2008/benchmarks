### To-do list


* windows cli (see pip install)  - YC

* new FE pipeline design
    * Dataset class (csv dataset, folder structure dataset, in-memory dataset, generator dataset)
        len(dataset)
        dataset[0]

    * integrate albumentation with Operator
    * fix pipeline with multi-processing


pipeline.benchmark()
it = pipeline.get_iterator(epoch=0, mode="train")
len(it)
it[0]

pipeline.transform()


network.show_results(pipeline=pipeline, epoch=0, mode="train")
network.transform()


* Version control in FE documentation page

* 