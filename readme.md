# Readme

Hello guys.

As you can see, data are prepared in the `./dataset`. The method to read the data is also ready. I'm building a toy VAE to learn the origin data. Just for fun.

This part collects any thoughts/plan about the main structure.

## prerequisites

We could maintain a doc to keep track of prerequisites. As far as I know, some packages we might need:

```
scanpy
scipy
geomloss
umap
optuna
numpy
sqlite
pandas
sklearn
torch
```


## Dataset

Our model use the schibinger dataset. This dataset is embryo sequencing profiles. We can use `scipy` to read the dataset. Sample already prepared to read it. For other dataset, some already contained in the scanpy.


## Structure model

I'm building a VAE. also we will test other VAE's performance.