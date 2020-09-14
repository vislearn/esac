#!/bin/bash

url=https://heidata.uni-heidelberg.de/api/access/datafile/:persistentId?persistentId=doi:10.11588/data/GSJE9D/9KTKH4
filename=12scenes.tar.gz

wget $url -O $filename
tar -xvzf $filename
rm $filename
