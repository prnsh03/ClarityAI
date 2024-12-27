#!/bin/bash

curl -L -o ffmpeg-git-amd64-static.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-git-amd64-static.tar.xz
sudo cp ffmpeg-git-*-amd64-static/ffmpeg /usr/local/bin/
sudo cp ffmpeg-git-*-amd64-static/ffprobe /usr/local/bin/
rm -r ffmpeg-git-*-amd64-static
rm ffmpeg-git-amd64-static.tar.xz
