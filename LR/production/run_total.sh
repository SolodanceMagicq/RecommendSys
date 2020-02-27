python="/usr/bin/python2.7"
origin_train_data="../data/train.txt"
origin_test_data="../data/test.txt"
train_data="../data/train_file"
test_data="../data/test_file"
lr_coef_file="../data/lr_coef"
lr_model_file="../data/lr_model"
feature_num_file="../data/feature_num"
if [ -f $origin_train_data -a -f $origin_test_data ]; then
    $python ana_train_data.py $origin_train_data $origin_test_data $train_data $test_data $feature_num_file
else
    echo "no origin file"
    exit
fi
if [ -f $train_data  ]; then
    $python train.py $train_data $lr_coef_file $lr_model_file $feature_num_file
else
    echo "no train file"
    exit
fi
if [ -f $lr_coef_file -a -f $lr_model_file ]; then
    $python check.py $test_data $lr_coef_file $lr_model_file $feature_num_file
else
    echo "no model file"
    exit
fi