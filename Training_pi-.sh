#! /bin/bash
######## Part 1 #########
# Script parameters     #
#########################
  
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
  
# Specify the QOS, mandatory option
#SBATCH --qos=normal
  
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
#SBATCH --account=mlgpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=e-
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/bes/mlgpu/xingty/myliu/log_train.out
#SBATCH --error=/hpcfs/bes/mlgpu/xingty/myliu/log_train.err
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --cpus-per-task=1  
#SBATCH --mem-per-cpu=40000
#
# Specify how many GPU cards to us:
#SBATCH --gres=gpu:v100:1
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
  
# list the allocated hosts
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
hostname
df -h
cd /hpcfs/bes/mlgpu/xingty/myliu
source /hpcfs/juno/junogpu/fangwx/setup_conda.sh
conda activate pytorch1.60 
which python
/usr/local/cuda/bin/nvcc --version
export workpath=/hpcfs/bes/mlgpu/xingty/myliu
echo $workpath


### --training ##
#python $workpath/dedx_flow.py --batch_size 128 --num_epochs 100 --training --num_block=3 --workpath $workpath -p=e-  -layer=0 --data_dir=$workpath/dataset/ --output_dir=$workpath --output_file='train.hdf5' --results_file=train.txt --save_model_name='e-.pth'
#python $workpath/dedx_flow.py --batch_size 128 --num_epochs 100 --training --num_block=3 --workpath $workpath -p=e-  --data_dir=$workpath/dataset/ --output_dir=$workpath --results_file=train.txt --save_model_name='e-.pth'
#python $workpath/dedx_flow.py --batch_size 128 --num_epochs 100 --training --restore --restore_file='pi-.pth' --num_block=3 --workpath $workpath -p=pi-  --data_dir=$workpath/dataset/ --output_dir=$workpath/results_pi-/ --results_file=train.txt --save_model_name='pi-_re.pth'
## --testing ##
#python $workpath/dedx_flow.py --generate_to_file --num_block=3 --restore_file='e-.pth' --gen_events=100  --gen_batch=100 --workpath $workpath -p=e-  --data_dir=$workpath/dataset/ --output_dir=$workpath --gen_dir=$workpath --output_file='test.hdf5' --results_file=test.txt
python dedx_flow.py --generate_to_file --num_block=3 --restore_file='e-.pth' --gen_events=10  --gen_batch=100 --workpath $workpath -p=e-  -layer=0  --data_dir=$workpath/dataset/ --output_dir=$workpath/ --gen_dir=$workpath/results/ --output_file='gen_train.hdf5' --results_file=gen.txt
#python $workpath/dedx_flow.py --generate_to_file --num_block=3 --restore_file='pi-_re.pth' --use_test_dataloader --gen_events=1000000  --gen_batch=100000 --workpath $workpath -p=pi-  --data_dir=$workpath/dataset/ --output_dir=$workpath/results_pi-/ --gen_dir=$workpath/results_pi-/gen/ --output_file='gen_test.hdf5' --results_file=gen.txt 
######### save pt ############
#python $workpath/dedx_flow.py --save_pt --num_block=3 --restore_file='pi-_re.pth' --workpath $workpath -p=pi-  --data_dir=$workpath/dataset/ --output_dir=$workpath/results_pi-/ --pt_file_path=$workpath/results_pi-/pi-_jit.pt
######### check pt ############
#python $workpath/dedx_flow.py --check_pt --workpath $workpath -p=pi-  --data_dir=$workpath/dataset/ --output_dir=$workpath/results_pi-/ --check_pt_file=$workpath/results_pi-/pi-_jit.pt
######### simulation ############
#python $workpath/dedx_flow.py --generate_to_file --num_block=3 --restore_file='pi-_re.pth' --gen_events=1000000  --gen_batch=4000 --num_try 5000  --workpath $workpath -p=pi-  --data_dir=$workpath/dataset/ --output_dir=$workpath/results_pi-/ --gen_dir=$workpath/results_pi-/sim/ --output_file='sim_train.hdf5' --results_file=sim.txt 

#python $workpath/dedx_flow.py --generate_to_file --restore_file='p+.pth' --gen_events=1000000  --gen_batch=100000 --workpath $workpath -p=p+  --data_dir=$workpath/dataset/ --output_dir=$workpath/results_p+/ --gen_dir=$workpath/results_p+/gen/ --output_file='gen_train.hdf5' --results_file=p+_gen.txt 
#python $workpath/dedx_flow.py --generate_to_file --restore_file='p+.pth' --use_test_dataloader --gen_events=1000000  --gen_batch=100000 --workpath $workpath -p=p+  --data_dir=$workpath/dataset/ --output_dir=$workpath/results_p+/ --gen_dir=$workpath/results_p+/gen/ --output_file='gen_test.hdf5' --results_file=p+_gen.txt 

### --training ##
#python $workpath/run_dedx_flow.py --training  --workpath $workpath -p=p+  --data_dir=$workpath/dataset/ --output_dir=$workpath/results/ --results_file=results_train.txt --save_model_name='rec_dedx_flow/p+_test_th1p6.pth'
## generating dE/dx distribution ##
#python $workpath/run_dedx_flow.py --generate_to_file --restore_file='rec_dedx_flow/p+.pt' --num_try 5000 --gen_batch 1000 --workpath $workpath -p=p+  --data_dir=$workpath/dataset/ --output_dir=$workpath/results/ --output_file='sim/p+_simdedxdist.hdf5' --results_file=results_simdedxdist.txt
#python $workpath/run_dedx_flow.py --data_dir=$workpath/dataset/  --save_pt  --restore_file='rec_dedx_flow/p+_test_th1p6.pth'  --workpath $workpath -p=p+   --pt_file_path=$workpath/p+_jit_th1p6.pt  --results_file=results_jit_forward.txt
#python $workpath/run_dedx_flow.py --data_dir=$workpath/dataset/  --check_pt  --check_pt_file=$workpath/p+_jit.pt --gen_batch 1 --workpath $workpath -p=p+  --results_file=results_jit.txt --output_dir=$workpath/results/ --output_file='p+_jitsimdedxdist.hdf5'
#python $workpath/check_pt_dedx_flow.py --data_dir=$workpath/dataset/  --check_pt  --check_pt_file=$workpath/p+_jit.pt  --workpath $workpath -p=p+  --results_file=results_jit.txt --output_dir=$workpath/results/ --output_file='p+_jitsimdedxdist.hdf5'

