#Please put here the path of images and TPS files from nonrigid benchmark
PATH_IMGS='../../eval_bench/All_PNG'
PATH_TPS='../../eval_bench/gt_tps'

#Set working dir to save results. Please change
working_dir='./nonrigid_eval'
mkdir -p $working_dir

#############################################################################

#Scripts Path
extract_gt_path='./extract_gt.py'
benchmark_path='./dalf_benchmark.py'
metrics_path='./plotUnorderedPR.py'

#Those names were used for the ablation study
ablation='eval_any'

#Data Path
network_path=$2

#Original TPS files
tps_dir_o=$PATH_TPS

#Local copy of TPS files
tps_dir=$working_dir'/gt_tps_'$ablation

#Output path
out_path=$working_dir'/out_'$ablation

echo 'copying original gt_tps '$tps_dir_o' to '$tps_dir
cp -rf $tps_dir_o $tps_dir
python3 $extract_gt_path -i $PATH_IMGS --tps_dir $tps_dir --dir -m pgdeal --net_path $network_path 
python3 $benchmark_path -i $PATH_IMGS -o $out_path --dir --sift --tps_path $tps_dir --net_path $network_path

# #Remove old results cache
rm *.dict

# #Show metric results
inputdir=$out_path

#Metric type: [MS, MMA, inliers]
metric=$1

python3 $metrics_path -i $inputdir/Kinect1 -d --tps_path $tps_dir --mode erase --metric $metric
python3 $metrics_path -i $inputdir/Kinect2Sampled -d --tps_path $tps_dir --mode append --metric $metric
python3 $metrics_path -i $inputdir/SimulationICCV -d --tps_path $tps_dir --mode erase --metric $metric
#python3 $metrics_path -i $inputdir/DeSurTSampled -d --tps_path $tps_dir --mode append --metric $metric

#Show stored final scores
python3 $metrics_path -i $inputdir/SimulationICCV -d --tps_path $tps_dir --mode append --metric $metric --gmean
