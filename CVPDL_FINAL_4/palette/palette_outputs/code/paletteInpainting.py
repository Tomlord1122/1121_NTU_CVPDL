# add path
import sys
sys.path.append('./Palette')
from Palette.run import main_worker
import Palette.core.praser as Praser
import Palette.core.util as Util
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json', help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-P', '--port', default='21012', type=str)


''' parser configs '''
args = parser.parse_args(['-c', './Palette/config/my_config.json', '-p', 'test'])
opt = Praser.parse(args)




# cuda device number (只支援單個 gpu，但是懶得改，所以請用[i])
opt['gpu_ids'] = [0]

# 原始照片資料夾（現在是會把資料夾裡面的所有照片都做）（要改再跟我說）
opt['datasets']['test']['which_dataset']['args']['data_root'] = './images'

# mask 資料夾 (mask)

opt['datasets']['test']['which_dataset']['args']['mask_config']['mask_root'] = './masks'

# 輸出會在 ./palette_outputs/results/test/0/Out_{原本相片名稱}.png


gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

opt['world_size'] = 1 
main_worker(0, 1, opt)