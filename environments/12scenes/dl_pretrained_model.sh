#!/bin/bash

url=https://cloudstore.zih.tu-dresden.de/index.php/s/LdnPbdCxoxJBnrH/download
filename=12scenes.tar.gz

wget $url -O $filename
tar -xvzf $filename
rm $filename
