#Override the default prompt, colorize $
PS1='${debian_chroot:+($debian_chroot)}\h:\w\[\033[1;36m\]$ \[\033[00m\]'

export PATH=$PATH:$HOME/.local/bin

#check if package is already installed before installing it
if ! python -c "import seismometer" 2> /dev/null; then
  pip install -r /home/seismo/workspace/requirements.txt
fi
