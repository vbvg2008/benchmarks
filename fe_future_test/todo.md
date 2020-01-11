
### to do list:



------------

6. add scheduling functionality

7. do benchmark on single gpu speed

8. add multi-gpu, do benchmark

* make it notebook friendly

## board

notebook usecase: ?

pipeline.benchmark()

pipeline.transform()

pipeline.show_batch()

estimator.fit, then fit again?



## Scheduler ideas:

things to schedule: querry dynamincally!

    train_data
    val_data
    ops
    batch_size

    optimizers

    ops - in Networks




still needs warm-up


 Update = reduce + update

 gradient_Op = reduce + gradient



Ideas 1:

users should provide the tf.data.dataset, torch.dataloader through pipeline

reason:
1. (tf.dataset, torch loader) users can benefit from pipeline functionalities like benchmark, show_batch ...


ideas 2:
no need to support a scheduler or pipelines,