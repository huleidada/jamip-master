#!/bin/bash

pip install psutil phonopy django==3.1 mendeleev matplotlib numpy scipy scikit-learn seaborn jinja2 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple 
python setup.py install 

# Add environment variable
if [[ $PATH != *$HOME/.jamip/bin* ]]; then
  echo 'export PATH=~/.jamip/bin:$PATH ' >> $HOME/.bashrc
fi
if [[ $PATH != *$HOME/.local/bin* ]]; then
  echo 'export PATH=~/.local/bin:$PATH ' >> $HOME/.bashrc
fi

# copy config directory 
if [ ! -d $HOME/.jamip/ ];then
  cp -r source ~/.jamip
  chmod +x -R ~/.jamip/bin/*
else
  echo "Config directory is exists."
fi
