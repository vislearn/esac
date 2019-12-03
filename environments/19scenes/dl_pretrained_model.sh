#!/bin/bash

url=https://cloudstore.zih.tu-dresden.de/index.php/s/RTBE2ZxL9g6zys5/download
filename=19scenes.tar.gz

wget $url -O $filename
tar -xvzf $filename
rm $filename
