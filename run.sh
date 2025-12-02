#!/bin/bash
set -xe

# Group configs by prefix
groups=(
    "g0:g0_rotary.py g0_transverse.py"
    "g2:g2_rotary.py g2_transverse.py"
    "ge:ge_rotary.py ge_transverse.py"
    "gg:gg_rotary.py gg_transverse.py"
)

CMD="python -m src.agents.ppo.train --logdir=logs/ --config=src/agents/ppo/configs"
TERMINAL="gnome-terminal"   # Change if needed

for group in "${groups[@]}"; do
    prefix="${group%%:*}"
    scripts="${group#*:}"

    echo "Launching terminal for group: $prefix"

    $TERMINAL -- bash -c "
        for cfg in $scripts; do
            echo Running \$cfg
            $CMD/\$cfg
            echo Done with \$cfg
        done
        echo All done for $prefix
        exec bash
    "
done
