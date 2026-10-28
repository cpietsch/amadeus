#!/bin/bash
# Stand-in for an unreliable `ssh musica`. Commands containing $FLAKY_MATCH fail like SSH ($FLAKY_MSG on stderr,
# exit 255) for the first $FLAKY_FAILS of them (counted in the file $FLAKY_COUNT). With FLAKY_PASS_BYTES=N the
# connection drops mid-transfer: the remote command first gets the first N bytes of stdin. Everything else goes to
# fake_ssh.sh unchanged.
here=$(cd "$(dirname "$0")" && pwd)
case $1 in
    *"$FLAKY_MATCH"*)
        n=$(( $(cat "$FLAKY_COUNT" 2>/dev/null || echo 0) + 1 ))
        echo "$n" > "$FLAKY_COUNT"
        if [ "$n" -le "${FLAKY_FAILS:-1000}" ]; then
            if [ -n "${FLAKY_PASS_BYTES:-}" ]; then
                head -c "$FLAKY_PASS_BYTES" | bash "$here/fake_ssh.sh" "$1" >/dev/null 2>&1
            fi
            echo "$FLAKY_MSG" >&2
            exit 255
        fi ;;
esac
exec bash "$here/fake_ssh.sh" "$@"
