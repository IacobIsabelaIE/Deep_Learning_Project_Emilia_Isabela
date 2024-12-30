from create_test_dataset import split_val_to_test
from config import DATASETgen

main_dirs = ['imagenet_ai_0419_biggan',
             'imagenet_ai_0419_vqdm',
             'imagenet_ai_0424_sdv5',
             'imagenet_ai_0424_wukong',
             'imagenet_ai_0508_adm',
             'imagenet_glide',
             'imagenet_midjourney']

split_val_to_test('tinygenimage', main_dirs)