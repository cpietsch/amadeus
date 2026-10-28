#!/bin/bash
# Stand-in for `ssh musica`: runs the remote command locally, with the fake cluster's HOME and Slurm stubs.
export HOME=$FAKE_HOME DATA=$FAKE_HOME/data SCRATCH=$FAKE_HOME/scratch USER=tester
export PATH="$(cd "$(dirname "$0")" && pwd)/fakebin:$PATH"
cd "$HOME" && exec bash -c "$1"
