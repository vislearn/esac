#!/bin/bash

url=https://cloudstore.zih.tu-dresden.de/index.php/s/8K7R9SnJPGNHy5y/download
filename=7scenes.tar.gz

wget $url -O $filename
tar -xvzf $filename
rm $filename
