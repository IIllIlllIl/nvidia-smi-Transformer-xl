#! /bin/bash

# Set default values
repetitions=1
thread=9

collect_energy_measurements()
{
  for i in $(seq 1 $2); do
    # Collect the energy consumption of the GPU
    nvidia-smi -i 0 --loop-ms=1000 --format=csv,noheader --query-gpu=power.draw,temperature.gpu,temperature.memory,utilization.gpu,utilization.memory >> nvidia_smi"$i".log &

    # Get nvidia-smi's PID
    nvidia_smi_PID=$!

    # Run model
    for j in $(seq 1 $3)
    do
    {
        $1 > resnet110_"$j".log
    }&
    done
    wait

    # When the experiment is elapsed, terminate the nvidia-smi process
    kill -9 "$nvidia_smi_PID"

    sleep 60

  done
}

sudo ./governor.sh pe

collect_energy_measurements "python3 -u trainer.py  --arch=resnet110  --save-dir=save_resnet110" "$repetitions" "$thread"


