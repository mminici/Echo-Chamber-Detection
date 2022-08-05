cd ../src
cuda_device='1'
annealing='constant'
model_type='softmax'
declare -a K=('4' '8' '16')
declare -a hyper_params=('1' '8' '16')
for k in "${K[@]}"; do
  for s in "${hyper_params[@]}"
  do
    for h in "${hyper_params[@]}"
    do
      for B in "${hyper_params[@]}"
      do
        python grid-real.py --annealing $annealing --K $k --s $s --h $h --B $B --model_type $model_type --device $cuda_device
      done
    done
  done
done
