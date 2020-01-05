
### to do list:


* make pytorch gpu training work with same speed  6000-7000

* make it notebook friendly

4. update several examples, let team know

6. add scheduling functionality

7. add single gpu, do benchmark on cretain gpu speed

8. add multi-gpu, do benchmark



Pipeline:
	- output filter mechanism, only output what's used by Network and Trace


Network:
	- input filter mechanism, only send network related data to gpu
	- output filter mechanism, only return Trace inputs from gpu to cpu - prediction

Estimator:
	batch (from cpu) prediction(from gpu)
