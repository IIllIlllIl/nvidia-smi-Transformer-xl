#!/bin/sh

# A script that executes all of our expriments
# and collects the required measurements

# Text fonts for Linux distros
bold=$(tput bold)
underline=$(tput smul)
default=$(tput sgr0)
greenlabel=$(tput setab 2)
redlabel=$(tput setab 1)
yellowlabel=$(tput setab 3)

# Set default values
repetitions=1
test_type="train"

# Help
help_info()
{
  echo "-r <repeitions number> or --repetitions <repeitions number> are used to define the number of repetitions to run each task"
  echo "-t <train | infer> or --test <train | infer> to perform the correspoding type of test"
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
# $1 is the name of the task (e.g., Transformer-XL)
# $2 is the name of the framework (e.g., PyTorch)
# $3 is the number of times to run each task
# $4 is the command to execute a task for the corresponding task ($1) that is written with a framework ($2)
collect_energy_measurements()
{
  log "Obtaining energy and run-time performance measurements"
 
  for i in $(seq 1 $3); do
    # Collect the energy consumption of the GPU
    nvidia-smi -i 0 --loop-ms=1000 --format=csv,noheader --query-gpu=power.draw,temperature.gpu,temperature.memory,utilization.gpu,utilization.memory >> "$measurements"/"$1"_"$2"_nvidia_smi_"$i".txt &

    # Get nvidia-smi's PID
    nvidia_smi_PID=$!
    
    # Collect the energy consumption of the processor package and main memory
    docker run --gpus all --init -it --rm --network=host --ipc=host -v $PWD:/workspace/transformer-xl transformer-xl bash run_wt103_base.sh train 1
    docker run --gpus all --init -it --rm --network=host --ipc=host -v $PWD:/workspace/transformer-xl transformer-xl bash run_wt103_base.sh train 1

    # Remove cached model
    if [ "$1" == "transformer_xl" ]; then
      rm data/wikitext-103/cache.pt
    fi

    if [ "$1" == "gnmt" ]; then
      rm -rf results
    fi

    # When the experiment is elapsed, terminate the nvidia-smi process
    kill -9 "$nvidia_smi_PID"

    log "Small sleep time to reduce power tail effecs"
    sleep 60
  
  done
}

# Get command-line arguments
OPTIONS=$(getopt -o r:t: --long repetitions:test -n 'run_experiments' -- "$@")
eval set -- "$OPTIONS"
while true; do
  case "$1" in
    -r|--repetitions) repetitions="$2"; shift 2;;
    -t|--test) test_type="$2"; shift 2;;
    -h|--help) help_info; shift;;
    --) shift; break;;
    *) >&2 log "${redlabel}[ERROR]${default} Wrong command line argument, please try again."; exit 1;;
  esac
done

# Switching to perfomrance mode
log "Switching to performance mode"
sudo ./governor.sh pe

# Test can be train or inference
if [ "$test_type" == "train" ]; then
  measurements="$PWD/measurements_train"
else
  measurements="$PWD/measurements_inference"
fi

[ -d "$measurements" ] && rm -rf "$measurements" && mkdir "$measurements"
[ ! -d "$measurements" ] && mkdir "$measurements"

# Go into the DeepLearning Examples repository
cd DeepLearningExamples

# Execute Transformer-XL for PyTorch and TensorFlow
log "Executing Transofrmer-XL for PyTorch"
cd PyTorch/LanguageModeling/Transformer-XL/

bash pytorch/scripts/docker/build.sh
if [ "$test_type" == "train" ]; then
  log "Training Transformer-XL for PyTorch"
  collect_energy_measurements "transformer_xl" "$i" "$repetitions" "docker run --gpus all --init -it --rm --network=host --ipc=host -v $PWD:/workspace/transformer-xl transformer-xl bash run_wt103_base.sh train 1"
else
  log "Testing Transformer-XL for PyTorch"
  collect_energy_measurements "transformer_xl" "$i" "$repetitions" "docker run --gpus all --init -it --rm --network=host --ipc=host -v $PWD:/workspace/transformer-xl transformer-xl bash run_wt103_base.sh eval 1"
fi
cd ../../../

log "Done with all tests"
return 0
