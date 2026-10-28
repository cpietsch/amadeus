#!/bin/bash
# Stand-in for `ssh <host>` to a client machine (e.g. the GPU workstation): runs the command locally with the fake
# host's HOME ($FAKE_HOST_HOME). With FAKE_HOST_TAR_FAIL=1, a command that writes a tar archive exits 1 after
# writing it (a producer that failed after sending a complete-looking archive).
export HOME=$FAKE_HOST_HOME
cd "$HOME" || exit 1
if [ "${FAKE_HOST_TAR_FAIL:-}" = 1 ]; then
    case $1 in *"tar czf"*) bash -c "$1"; echo "tar: some file: Cannot stat: No such file or directory" >&2; exit 1 ;; esac
fi
exec bash -c "$1"
