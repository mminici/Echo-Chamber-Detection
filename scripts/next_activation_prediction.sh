cd ../src
cuda_device='3'
removing_prop='True'
model_type='exact-posterior'
declare -a hyper_params=('0.05' '0.1' '0.15' '0.20' '0.25' '0.30')
declare -a datasets=('vaxNoVax' 'referendum' 'brexit')
for dataset_name in "${datasets[@]}";
do
  for prop_masking_perc in "${hyper_params[@]}";
    do
      python grid-real.py --model_type $model_type --device $cuda_device --dataset $dataset_name --prop_removal_perc $prop_masking_perc --prop_removal $removing_prop
    done
done

