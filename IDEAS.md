# Idea For Apairo


## API 

Apairo an api to easily describe from the terminal the current state of your dataset.

```sh
apairo /data/rellis as RellisDataset
>>> Not intialized yiet
...
```

Or something more like git
```sh
cd /data/rellis
apairo init as RellisDataset
>>> Rellis already init
apairo describe
>>> Raw installed : ['lidar', 'poses', 'labels']
>>> Preprocess : ['trav_traj']
>>> Sequences : ['00000', '00001', '00002']
>>> Number of frame : 14521
...
apairo scan
>>> New preprocess detected : 'trav_gt'
>>> Save in apairo? (y/N) y
>>> Name (def: trav_gt) : 
>>> From : lidar
>>> trav_gt added
```




