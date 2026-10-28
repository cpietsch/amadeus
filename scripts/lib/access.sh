# Sourced by scripts/mx and scripts/musica_auth.sh: the same SSH config, network path and ASC username as Amadeus.
# The server passes MUSICA_SSH_CONFIG, MUSICA_VPN and MUSICA_ASC_USER; run by hand, amadeus/network.py works them out
# (and writes the ssh config), so both sides always decide the same way.
#   SSHF: options for ssh: -F <Amadeus's ssh config>, or none when ~/.ssh/config has its own `musica` entry

if [ -z "${MUSICA_SSH_CONFIG+x}" ]; then
    _root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    if _a=$(uv run --quiet --project "$_root" python -m amadeus.network 2>/dev/null); then
        eval "$_a"
    else
        MUSICA_SSH_CONFIG=     # no uv: plain `ssh musica` (an own entry in ~/.ssh/config)
    fi
fi
SSHF=()
[ -n "$MUSICA_SSH_CONFIG" ] && SSHF=(-F "$MUSICA_SSH_CONFIG")

access_vpn() {
    case "${MUSICA_VPN:-}" in wireproxy|system) echo "$MUSICA_VPN" ;;
        *) [ -f ~/.config/wireguard/wireproxy.conf ] && echo wireproxy || echo system ;;
    esac
}
