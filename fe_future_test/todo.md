
### to do list:

* make it notebook friendly

------------

6. add scheduling functionality

7. do benchmark on single gpu speed

8. add multi-gpu, do benchmark



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
