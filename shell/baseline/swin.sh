#!/bin/sh

# A script that executes all of our expriments
# and collects the required measurements

# Set default values
repetitions=1
epoch=200
lr=0.1
weight=0.0001

# Help
help_info()
{
  echo "-r <repeitions number> or --repetitions <repeitions number> are used to define the number of repetitions to run each task"
  exit
}

# Log with a timestamp
log()
{
  # Output is redirected to the log file if needed at the script's lop level
  date +'%F %T ' | tr -d \\n 1>&2
  echo "$@" 1>&2
}

# Function that executes
collect_energy_measurements()
{
  log "Obtaining energy and run-time performance measurements"
 
  for i in $(seq 1 $2); do  
    # Collect the energy consumption of the GPU
    nvidia-smi --loop-ms=1000 --format=csv,noheader --query-gpu=power.draw,temperature.gpu,temperature.memory,utilization.gpu,utilization.memory >> nvidia_smi"$i".log &

    # Get nvidia-smi's PID
    nvidia_smi_PID=$!

    # Run model
    perf stat -e power/energy-pkg/,power/energy-ram/ $1>out.log 2>> ./cpu.log

    # When the experiment is elapsed, terminate the nvidia-smi process
    kill -9 "$nvidia_smi_PID"

    log "Small sleep time to reduce power tail effecs"
    sleep 60

  done
}

# Get command-line arguments
OPTIONS=$(getopt -o r:e:l:w: --long repetitions:test -n 'run_experiments' -- "$@")
eval set -- "$OPTIONS"
while true; do
  case "$1" in
    -r|--repetitions) repetitions="$2"; shift 2;;
    -e|--epoch) epoch="$2"; shift 2;;
    -l|--lr)lr="$2";shift 2;;
    -w|--wd)weight="$2";shift 2;;
    -h|--help) help_info; shift;;
    --) shift; break;;
    *) >&2 log "${redlabel}[ERROR]${default} Wrong command line argument, please try again."; exit 1;;
  esac
done

# Switching to perfomrance mode
log "Switching to performance mode"
sudo ./governor.sh pe

log "epoch: "$epoch""
collect_energy_measurements "python train.py --use_swin --name swin --weight-decay="$weight" --learning-rate="$lr" --epochs="$epoch"" "$repetitions"

log "Done with all tests"


