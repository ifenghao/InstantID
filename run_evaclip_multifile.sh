export HF_HOME=~/huggingface

image_path=./data/portrait_train
save_root=./data/portrait_train_evaclip

USE_NUM=8

oldIFS=$IFS
IFS=$'\n'

templates=(`ls -d ${image_path}/accept_arrow_data*txt | shuf`)

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
        ((gpu = i % 8))
        echo ${temps}---process:${i}---gpu:${gpu}
        CUDA_VISIBLE_DEVICES=$i python3 evaclip.py \
            --image_path_files="${temps}" \
            --save_root="${save_root}"
    } &
done
wait

IFS=$oldIFS
