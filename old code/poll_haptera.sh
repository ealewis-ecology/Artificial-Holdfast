#!/usr/bin/env bash
# Poll the EC2 host for haptera_export.py completion, download the STL, and
# trigger a macOS notification. Run locally on the Mac.
#
#   ./poll_haptera.sh                                 # foreground
#   nohup ./poll_haptera.sh > poll.log 2>&1 &         # background, survives terminal close

set -u

# ── EDIT THESE ─────────────────────────────────────────────────────────────
SSH_USER="ec2-user"                                   # "ubuntu" on Ubuntu AMIs
SSH_HOST="18.118.107.231"
SSH_KEY="/Users/elile/Documents/MLML/Thesis/Artificial Holdfast/Eli's MacBook.pem"
REMOTE_DIR="~/Artificial-Holdfast"                    # dir where you ran the script
LOCAL_DIR="/Users/elile/Documents/MLML/Thesis/Artificial Holdfast"
STL_NAME="haptera_d9_k2_r130_h130_f650.stl"           # matches current config in haptera_export.py
INTERVAL=120                                          # seconds between polls

# Shutdown after successful download. Tries methods in order:
#   1. Local `aws ec2 ...-instances` (needs AWS CLI + creds locally)
#   2. Remote `aws` via SSH (needs an IAM role attached to the instance)
#   3. `sudo shutdown -h +1` via SSH (always works; STOPS the instance, doesn't terminate)
INSTANCE_ID="i-0d4374d1327c75297"
AWS_REGION="us-east-2"
SHUTDOWN_ACTION="terminate"                           # "terminate" / "stop" / "" (skip)
# ───────────────────────────────────────────────────────────────────────────

SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30)

notify() {
    osascript -e "display notification \"$2\" with title \"$1\" sound name \"Glass\""
}

SHUTDOWN_RESULT=""

shutdown_instance() {
    if [ -z "$SHUTDOWN_ACTION" ]; then
        SHUTDOWN_RESULT="skipped"
        return 0
    fi

    # Attempt 1: local AWS CLI
    if command -v aws >/dev/null 2>&1; then
        echo "[$(date +%T)] attempting ${SHUTDOWN_ACTION} via local AWS CLI..."
        if aws ec2 "${SHUTDOWN_ACTION}-instances" --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" >/dev/null 2>&1; then
            SHUTDOWN_RESULT="${SHUTDOWN_ACTION}d (local aws)"
            return 0
        fi
        echo "[$(date +%T)] local AWS CLI ${SHUTDOWN_ACTION} failed (creds/perms?)"
    fi

    # Attempt 2: remote AWS CLI via instance IAM role
    echo "[$(date +%T)] attempting ${SHUTDOWN_ACTION} via EC2's own AWS CLI..."
    if ssh "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" "aws ec2 ${SHUTDOWN_ACTION}-instances --instance-ids $INSTANCE_ID --region $AWS_REGION" >/dev/null 2>&1; then
        SHUTDOWN_RESULT="${SHUTDOWN_ACTION}d (instance role)"
        return 0
    fi
    echo "[$(date +%T)] remote ${SHUTDOWN_ACTION} failed (no IAM role with ec2 perms)"

    # Attempt 3: sudo shutdown via SSH — only stops, doesn't terminate
    echo "[$(date +%T)] falling back to 'sudo shutdown -h +1' via SSH (will STOP, not terminate)..."
    if ssh "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" "sudo shutdown -h +1" >/dev/null 2>&1; then
        SHUTDOWN_RESULT="stopped — terminate manually at https://console.aws.amazon.com/ec2/home?region=$AWS_REGION#Instances:instanceId=$INSTANCE_ID"
        return 0
    fi

    SHUTDOWN_RESULT="all shutdown attempts FAILED — instance still running"
    return 1
}

echo "[$(date +%T)] polling $SSH_HOST every ${INTERVAL}s for $STL_NAME"

SSH_FAIL_STREAK=0
SSH_FAIL_LIMIT=10  # ~20 min at INTERVAL=120 before giving up

while true; do
    status=$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" "
        if pgrep -f haptera_export.py >/dev/null 2>&1; then
            echo RUNNING
        elif [ -s $REMOTE_DIR/$STL_NAME ]; then
            echo DONE:\$(stat -c%s $REMOTE_DIR/$STL_NAME)
        else
            echo CRASHED
        fi
    " 2>/dev/null)
    ssh_exit=$?

    if [ $ssh_exit -ne 0 ] || [ -z "$status" ]; then
        SSH_FAIL_STREAK=$((SSH_FAIL_STREAK + 1))
        echo "[$(date +%T)] SSH check failed (exit=$ssh_exit, streak=$SSH_FAIL_STREAK/$SSH_FAIL_LIMIT) — retrying in ${INTERVAL}s"
        if [ $SSH_FAIL_STREAK -ge $SSH_FAIL_LIMIT ]; then
            notify "haptera_export poller giving up" "$SSH_FAIL_LIMIT consecutive SSH failures — instance unreachable"
            exit 1
        fi
        sleep "$INTERVAL"
        continue
    fi
    SSH_FAIL_STREAK=0

    case "$status" in
        RUNNING)
            sleep "$INTERVAL"
            ;;
        DONE:*)
            size=${status#DONE:}
            echo "[$(date +%T)] run complete: STL is ${size} bytes. downloading..."
            if ! scp "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/$STL_NAME" "$LOCAL_DIR/"; then
                echo "[$(date +%T)] STL download FAILED — leaving instance running so you can retry"
                notify "haptera_export download failed" "STL not retrieved — instance left running"
                exit 1
            fi
            scp "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/${STL_NAME%.stl}.txt" "$LOCAL_DIR/" 2>/dev/null || true

            shutdown_instance || true
            echo "[$(date +%T)] shutdown result: $SHUTDOWN_RESULT"
            notify "haptera_export done" "$STL_NAME downloaded — $SHUTDOWN_RESULT"
            echo "[$(date +%T)] done — files in $LOCAL_DIR"
            exit 0
            ;;
        CRASHED)
            echo "[$(date +%T)] process gone and STL missing — likely crashed. check run.log on the host."
            notify "haptera_export FAILED" "process exited but STL missing — check run.log on EC2"
            exit 1
            ;;
        *)
            echo "[$(date +%T)] unexpected status: '$status' — retrying"
            sleep "$INTERVAL"
            ;;
    esac
done
