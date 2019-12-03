#!/bin/bash

url=https://cloudstore.zih.tu-dresden.de/index.php/s/WGPtxD8yNX7mxyw/download
filename=dubrovnik.tar.gz

wget $url -O $filename
tar -xvzf $filename
rm $filename
