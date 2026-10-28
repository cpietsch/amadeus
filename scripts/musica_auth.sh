#!/bin/bash
# Make sure we can SSH to MUSICA non-interactively (Linux and macOS):
#   1. the network path: MUSICA_VPN=wireproxy starts the userspace WireGuard tunnel (containers without a TUN device,
#      scripts/vpn.sh); MUSICA_VPN=system leaves it to the operating system (WireGuard app, university network).
#      Default auto: as amadeus/network.py decides it (scripts/lib/access.sh).
#   2. an ssh-agent: the IdentityAgent of the `musica` host in Amadeus's ssh config (or an own entry in ~/.ssh/config):
#      a dedicated socket, started here when it is not running (needed for --renew); else the agent in $SSH_AUTH_SOCK.
#   3. a valid step-ssh certificate (~16 h, issued via auth.asc.ac.at) in that agent.
#      If not, start `step ssh login --console` in the background and print the device-flow link; step polls until
#      the user approves it in a browser. The grant expires after about 60 s: approve right away.
#   A certificate with less than 60 min left counts as expiring: a new login starts, so a renewal can be done
#   before it lapses.
# Usage: scripts/musica_auth.sh [--wait]
#        scripts/musica_auth.sh --renew     # new certificate now, while the current one is still valid (waits; see below)
#        scripts/musica_auth.sh --check     # only print the expiry and minutes left (no login, no side effects);
#                                           # exit 1 if < 60 min

HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/lib/access.sh"
SSH_HOST=${MUSICA_SSH_HOST:-musica}
# ASC username: $MUSICA_USER, else Amadeus's [cluster] user (MUSICA_ASC_USER), else the User of an own `musica` entry
ASC_USER=${MUSICA_USER:-${MUSICA_ASC_USER:-$(ssh "${SSHF[@]}" -G "$SSH_HOST" 2>/dev/null | awk '/^user /{print $2}')}}
LOG=~/.step/asc-login.log
mkdir -p ~/.step

IA=$(ssh "${SSHF[@]}" -G "$SSH_HOST" 2>/dev/null | awk '/^identityagent /{sub(/^identityagent /, ""); print}')
case "$IA" in
    ""|none|SSH_AUTH_SOCK) DEDICATED=0 ;;
    *) DEDICATED=1; IA=${IA/#\~/$HOME}; export SSH_AUTH_SOCK=$IA ;;
esac

to_epoch() {   # ISO time (local) -> epoch seconds, with GNU date, BSD date or Python
    date -d "$1" +%s 2>/dev/null || date -j -f '%Y-%m-%dT%H:%M:%S' "$1" +%s 2>/dev/null ||
        python3 -c 'import sys, datetime; print(int(datetime.datetime.fromisoformat(sys.argv[1]).timestamp()))' "$1"
}

cert_expiry() {
    step ssh list --raw 2>/dev/null | while read -r line; do
        echo "$line" | ssh-keygen -L -f - 2>/dev/null | sed -n 's/.*Valid: from .* to //p'
    done | sort | tail -1
}

if [ "${1:-}" = "--check" ]; then
    exp=$(cert_expiry)
    [ -z "$exp" ] && { echo "no certificate"; exit 1; }
    left=$(( ($(to_epoch "$exp") - $(date +%s)) / 60 ))
    echo "certificate valid until $exp ($left min left)"
    [ "$left" -ge 60 ]
    exit
fi

if [ "$(access_vpn)" = wireproxy ]; then
    pgrep -f "wireproxy -c" >/dev/null || "$HERE/vpn.sh" up || exit 1
fi

ssh-add -l >/dev/null 2>&1
if [ $? = 2 ]; then
    [ "$DEDICATED" = 1 ] || { echo "no ssh-agent reachable (SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-unset}); start one, or let Amadeus write its ssh config (start the server once)"; exit 1; }
    rm -f "$SSH_AUTH_SOCK"
    ssh-agent -a "$SSH_AUTH_SOCK" >/dev/null || exit 1
fi

exp=$(cert_expiry)
# --renew: step refuses a new login while a valid certificate is in the agent ("already present"), so log in
# through a second agent and swap it in after approval; access keeps working until then. Needs the dedicated socket.
if [ "${1:-}" = "--renew" ]; then
    if [ "$DEDICATED" != 1 ]; then
        echo "login failed: early renewal needs a dedicated agent socket (the IdentityAgent of Amadeus's ssh config); log in again once the certificate has expired"
        exit 1
    fi
    NEW=$SSH_AUTH_SOCK.renew; RLOG=~/.step/asc-renew.log
    rm -f "$NEW"; ssh-agent -a "$NEW" > "$NEW.env" || exit 1
    NEWPID=$(sed -n 's/^SSH_AGENT_PID=\([0-9]*\).*/\1/p' "$NEW.env")
    SSH_AUTH_SOCK=$NEW step ssh login "$ASC_USER" --context asc --console > "$RLOG" 2>&1 < /dev/null &
    for _ in 1 2 3 4 5 6 7 8 9 10; do grep -q 'device?code' "$RLOG" && break; sleep 1; done
    echo "Renewing (current certificate valid until ${exp:-none}). Approve in a browser within about 60 s:"
    grep -o 'https://auth.asc.ac.at/device?code=[0-9]*' "$RLOG" | tail -1
    wait
    if SSH_AUTH_SOCK=$NEW ssh-add -l 2>/dev/null | grep -q CERT; then
        for p in $(pgrep -x ssh-agent); do      # the old agent serving this socket
            [ "$p" != "$NEWPID" ] && ps -o args= -p "$p" 2>/dev/null | grep -qE -- "-a ($SSH_AUTH_SOCK|$HOME/.ssh/agent-renew.sock)" && kill "$p"
        done
        mv -f "$NEW" "$SSH_AUTH_SOCK"; rm -f "$NEW.env"
        echo "certificate valid until $(cert_expiry)"
    else
        kill "$NEWPID"; rm -f "$NEW" "$NEW.env"; echo "login failed:"; cat "$RLOG"; exit 1
    fi
    exit 0
fi

if [ -n "$exp" ] && [ "$(to_epoch "$exp")" -gt "$(( $(date +%s) + 3600 ))" ]; then
    echo "certificate valid until $exp"
    exit 0
fi

if ! pgrep -f "step ssh login $ASC_USER" >/dev/null; then
    nohup step ssh login "$ASC_USER" --context asc --console >"$LOG" 2>&1 </dev/null &
    sleep 5
fi
echo "No valid certificate. Approve in a browser:"
grep -o 'https://auth.asc.ac.at/device?code=[0-9]*' "$LOG" | tail -1

if [ "${1:-}" = "--wait" ]; then
    while pgrep -f "step ssh login $ASC_USER" >/dev/null; do sleep 3; done
    exp=$(cert_expiry)
    [ -n "$exp" ] && echo "certificate valid until $exp" || { echo "login failed:"; cat "$LOG"; exit 1; }
fi
