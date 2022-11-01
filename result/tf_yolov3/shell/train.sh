#!/bin/sh

# A script that executes all of our expriments
# and collects the required measurements

# Set default values
repetitions=1
thread=1

# Help
help_info()
{
  echo "-r <repeitions number> or --repetitions <repeitions number> are used to define the number of repetitions to run each task"
  exit
}

collect()
{
        mkdir ./cgt"$1"
        mv ./*.log ./cgt"$1"
        mv ./cgt"$1" ./result/
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
    nvidia-smi -i 0 --loop-ms=1000 --format=csv,noheader --query-gpu=power.draw,temperature.gpu,temperature.memory,utilization.gpu,utilization.memory >> nvidia_smi"$i".log &

    # Get nvidia-smi's PID
    nvidia_smi_PID=$!

    # Run model
    for j in $(seq 0 $3)
    do
    {
        $1 > "$j".log
    }&
    done
    wait

    # When the experiment is elapsed, terminate the nvidia-smi process
    kill -9 "$nvidia_smi_PID"

    log "Small sleep time to reduce power tail effecs"
    sleep 60

    collect "$3"

  done
}

# Get command-line arguments
OPTIONS=$(getopt -o r:t: --long repetitions:test -n 'run_experiments' -- "$@")
eval set -- "$OPTIONS"
while true; do
  case "$1" in
    -r|--repetitions) repetitions="$2"; shift 2;;
    -t|--thread) thread="$2"; shift 2;;
    -h|--help) help_info; shift;;
    --) shift; break;;
    *) >&2 log "${redlabel}[ERROR]${default} Wrong command line argument, please try again."; exit 1;;
  esac
done

# Switching to perfomrance mode
log "Switching to performance mode"
sudo ./governor.sh pe

# Execute Transformer-XL for PyTorch and TensorFlow
collect_energy_measurements "python3 ../train.py" "$repetitions" "$thread"

log "Done with all tests"
return 0
