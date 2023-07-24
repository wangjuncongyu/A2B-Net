import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
add_path(osp.join(this_dir, '..'))


add_path(osp.join(this_dir, '..', 'datasets'))
add_path(osp.join(this_dir, '..', 'utils'))
add_path(osp.join(this_dir, '..', 'models'))
# Add util to PYTHONPATH
#utils_path = osp.join(this_dir, '..', 'utils')
#add_path(utils_path)
