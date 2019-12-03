### Reproducing Our Main Results for 19Scenes

Install the ESAC c++ extension (if not done before):

```
/code/esac> python setup.py install
```

Setup the datasets (if not done before):

```
/datasets> python setup_7scenes.py
/datasets> python setup_12scenes.py
```

Train ESAC:

```
/environments/19scenes> python ../../code/init_gating.py 
/environments/19scenes> python ../../code/init_expert.py -e <0 to 18>
/environments/19scenes> python ../../code/train_esac.py
```

Test ESAC:
```
/environments/19scenes> python ../../code/test_esac.py
```

Test pre-trained models:
```
/environments/19scenes> sh dl_pretrained_model.sh
/environments/19scenes> python ../../code/test_esac.py -sid pretrained
```
