#!/bin/bash
# Start/stop the userspace WireGuard tunnel (wireproxy) to ASC.
# The container has no NET_ADMIN/TUN, so wg-quick can't be used; wireproxy
# exposes the tunnel as a SOCKS5 proxy on 127.0.0.1:1080 that the ssh config's ProxyCommand uses.
# Usage: scripts/vpn.sh up|down|status

CONF=~/.config/wireguard/wireproxy.conf
LOG=~/.config/wireguard/wireproxy.log
BIN=$(command -v wireproxy 2>/dev/null || echo ~/.local/bin/wireproxy)

running() { pgrep -f "wireproxy -c $CONF" >/dev/null; }

case "${1:-status}" in
    up)
        [ -f ~/.config/wireguard/asc.conf ] || { echo "missing ~/.config/wireguard/asc.conf"; exit 1; }
        running && { echo "already running"; exit 0; }
        "$BIN" -n -c "$CONF" || exit 1          # config check
        nohup "$BIN" -c "$CONF" >"$LOG" 2>&1 &
        sleep 2
        running && echo "wireproxy up (SOCKS5 127.0.0.1:1080), log: $LOG" || { echo "failed:"; cat "$LOG"; exit 1; }
        ;;
    down)
        pkill -f "wireproxy -c $CONF" && echo "stopped" || echo "not running"
        ;;
    status)
        running && echo "running" || echo "stopped"
        for h in musica.vie.asc.ac.at musica.lnz.asc.ac.at musica.inn.asc.ac.at; do
            printf "%-24s " "$h"
            nc -X 5 -x 127.0.0.1:1080 -w 8 "$h" 22 </dev/null 2>/dev/null | head -1 | grep . || echo "unreachable"
        done
        ;;
    *) echo "usage: $0 up|down|status"; exit 2 ;;
esac
