#!/bin/bash
# run_evals_shared_display.sh
# All processes share one Xvfb display (since --noshow_gui is used)

GAITS=(
    "g2_rotary_no_rai"
    "g0_rotary_no_rai"
    "gg_rotary_no_rai"
    "ge_rotary_no_rai"
    "g2_transverse_no_rai"
    "g0_transverse_no_rai"
    "gg_transverse_no_rai"
    "ge_transverse_no_rai"
)

# Start ONE Xvfb server that all processes will share
echo "Starting shared Xvfb display :99..."
Xvfb :99 -screen 0 1920x1080x24 > /dev/null 2>&1 &
XVFB_PID=$!
echo "Xvfb PID: $XVFB_PID"
echo ""

sleep 3  # Let Xvfb start

echo "Launching all evaluations on shared display :99..."
echo ""

GPU=0
PIDS=()

for gait in "${GAITS[@]}"; do
    FOLDER=$(ls -td check_pt/${gait}/*/ 2>/dev/null | head -1)
    
    if [ -z "$FOLDER" ]; then
        echo "⚠️  No folder found for $gait, skipping..."
        GPU=$((GPU + 1))
        continue
    fi
    
    FOLDER="${FOLDER%/}"
    
    echo "🚀 [$GPU] $gait"
    echo "   📁 $FOLDER"
    
    # All use DISPLAY=:99
    CUDA_VISIBLE_DEVICES=$GPU \
    DISPLAY=:99 \
    nohup python -m src.agents.ppo.eval_cot5 \
      --record_video \
      --noshow_gui \
      --logdir="$FOLDER" \
      > "eval_${gait}.log" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    echo "   PID: $PID"
    echo ""
    
    GPU=$((GPU + 1))
    sleep 10  # Stagger initialization
done

echo "================================================"
echo "✓ All launched! Running in parallel on display :99"
echo "================================================"
echo ""
echo "Xvfb PID: $XVFB_PID"
echo "Eval PIDs: ${PIDS[@]}"
echo ""
echo "Monitor:"
echo "  tail -f eval_g2_rotary_no_rai.log"
echo "  ps aux | grep eval_cot5 | grep -v grep | wc -l  # Count running"
echo ""
echo "Kill all:"
echo "  pkill -f eval_cot5  # Kill evals"
echo "  kill $XVFB_PID      # Kill Xvfb"
echo ""
echo "Or kill everything:"
echo "  pkill -f eval_cot5 && pkill Xvfb"
