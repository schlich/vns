#!/bin/bash

# Remove existing Jupyter kernels
rm -rf /root/.local/share/jupyter/kernels

# Install Nix
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install linux --init none --no-confirm
ln -s /root/.nix-profile/bin /opt/bin

# Install evcxr
nix-env -f '<nixpkgs>' -iA cargo rustc evcxr sccache
evcxr_jupyter --install

# Configure evcxr
mkdir -p /root/.config/evcxr
printf ":timing\n:sccache 1" > /root/.config/evcxr/init.evcxr

# Download and install IPC Proxy
wget -qO- https://gist.github.com/wiseaidev/bc102165f43db4ebd84fcdb4c5bfb129/archive/b087c21310402bc999b36fecaf63207c74cf5b90.tar.gz | tar xvz --strip-components=1
python install_ipc_proxy_kernel.py --quiet --kernel=rust --implementation=ipc_proxy_kernel.py > /dev/null

# Update kernel display name
sed -i 's/"display_name": "Rust"/"display_name": "Rust-TCP"/g' /root/.local/share/jupyter/kernels/rust_tcp/kernel.json

# Restart Jupyter notebook
bash -c "killall jupyter-notebook ; sleep 3 ; source ~/.cargo/env && jupyter notebook --ip=172.28.0.12 --port=9000" </dev/null>/dev/null &
