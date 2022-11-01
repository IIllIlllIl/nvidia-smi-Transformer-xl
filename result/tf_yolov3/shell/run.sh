#! /bin/bash

# Set default values
repetitions=1
thread=1

collect_energy_measurements()
{
  log "Obtaining energy and run-time performance measurements"

  for i in $(seq 1 $2); do
    # Collect the energy consumption of the GPU
    nvidia-smi -i 0 --loop-ms=1000 --format=csv,noheader --query-gpu=power.draw,temperature.gpu,temperature.memory,utilization.gpu,utilization.memory >> nvidia_smi"$i".log &

    # Get nvidia-smi's PID
    nvidia_smi_PID=$!

    # Run model
    for j in $(seq 1 $3)
    do
    {
        $1
    }&
    done
    wait

    # When the experiment is elapsed, terminate the nvidia-smi process
    kill -9 "$nvidia_smi_PID"

    log "Small sleep time to reduce power tail effecs"
    sleep 60

  done
}

sudo ./governor.sh pe

collect_energy_measurements "python3 train.py" "$repetitions" "$thread"


