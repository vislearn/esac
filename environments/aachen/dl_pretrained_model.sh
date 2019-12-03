#!/bin/bash

url=https://cloudstore.zih.tu-dresden.de/index.php/s/krszHYbHetxjmqp/download
filename=aachen.tar.gz

wget $url -O $filename
tar -xvzf $filename
rm $filename
