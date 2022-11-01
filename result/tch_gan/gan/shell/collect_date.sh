#! /bin/bash

thread=1

collect()
{
	mkdir ./t"$1"
	mv ./*.log ./t"$1"
	mv ./nvidia_smi1.txt ./t"$1"
	mv ./t"$1" ./result/
}

thread=9
collect "$thread"

