arrow_root=./*_data/
path_root=./training_pmv2_face_filelist
save_root=./data/portrait_dataset
image_root=./data/portrait_dataset

USE_NUM=32

oldIFS=$IFS
IFS=$'\n'
# templates=(`ls -d ${arrow_root}/*.arrow`)
templates=(`ls -d ${path_root}/*.txt`)
echo ${#templates[@]}
((inference_num=(${#templates[@]} - 1)/${USE_NUM} + 1))
template_splits=()
template_name=""
cnt=1
for temp in ${templates[@]}
do
    if [ $cnt -eq 1 ] 
    then
        template_name="$temp"
    else
        template_name+=",$temp"
    fi
    if [ $cnt -ge $inference_num ]
    then
        template_splits+=($template_name)
        template_name=""
        cnt=1
    else
        ((cnt += 1))
    fi
done
if [ $cnt -ne 1 ]
then
    template_splits+=($template_name)
fi

for ((i=0;i<${#template_splits[@]};i++));
do
    {
        temps=${template_splits[i]}
        echo ${temps}---process:${i}
        # python3 dataset.py \
        #     --arrow_files="${temps}" \
        #     --image_root="${image_root}" \
        #     --save_root="${save_root}"
        python3 dataset.py \
            --path_files="${temps}" \
            --save_root="${save_root}"
    } &
done
wait

IFS=$oldIFS
