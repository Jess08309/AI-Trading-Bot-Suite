#!/bin/bash
set -euo pipefail

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    echo "This script must be run as root (use sudo)."
    exit 1
fi

for svc in putseller spreadbot alpacabot callbuyer; do
    for ext in service timer; do
        unit="${svc}.${ext}"
        systemctl disable --now "$unit" 2>/dev/null || true
        rm -f "/etc/systemd/system/${unit}" "/lib/systemd/system/${unit}" "/usr/lib/systemd/system/${unit}"
        for link_dir in /etc/systemd/system/*.wants /etc/systemd/system/*.requires; do
            [ -d "$link_dir" ] || continue
            rm -f "$link_dir/$unit"
        done
        systemctl reset-failed "$unit" 2>/dev/null || true
    done
done

systemctl daemon-reload
