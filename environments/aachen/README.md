### Reproducing Our Main Results for Aachen

Install the ESAC c++ extension (if not done before):

```
/code/esac> python setup.py install
```

Setup the dataset (if not done before):

```
/datasets> python setup_aachen.py
```

Train ESAC (with 10 experts):

```
/environments/aachen> python ../../code/init_gating.py -c 10 -it 5000000
/environments/aachen> python ../../code/init_expert.py -c 10 -e <0 to 9>
/environments/aachen> python ../../code/ref_expert.py -c 10 -e <0 to 9>
/environments/aachen> python ../../code/train_esac.py -c 10 -ref -maxe 5
```

Test ESAC (with 10 experts):
```
/environments/aachen> python ../../code/test_esac.py -c 10
```
**Note:** Since there is no public ground truth, the test script will display arbitrary pose errors. Upload the poses_*.txt output file to [https://www.visuallocalization.net/](https://www.visuallocalization.net/) to calculate error metrics.

Test pre-trained models (with 10/20/50 experts):
```
/environments/aachen> sh dl_pretrained_model.sh
/environments/aachen> python ../../code/test_esac.py -c 10 -sid c10_pretrained
/environments/aachen> python ../../code/test_esac.py -c 20 -sid c20_pretrained
/environments/aachen> python ../../code/test_esac.py -c 50 -sid c50_pretrained
```

