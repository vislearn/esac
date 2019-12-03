### Reproducing Our Main Results for 12Scenes

Install the ESAC c++ extension (if not done before):

```
/code/esac> python setup.py install
```

Setup the dataset (if not done before):

```
/datasets> python setup_12scenes.py
```

Train ESAC:

```
/environments/12scenes> python ../../code/init_gating.py 
/environments/12scenes> python ../../code/init_expert.py -e <0 to 11>
/environments/12scenes> python ../../code/train_esac.py
```

Test ESAC:
```
/environments/12scenes> python ../../code/test_esac.py
```

Test pre-trained models:
```
/environments/12scenes> sh dl_pretrained_model.sh
/environments/12scenes> python ../../code/test_esac.py -sid pretrained
```
