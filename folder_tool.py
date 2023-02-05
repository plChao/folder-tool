import os
import glob
import sys
import json

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        print('create not exists folder', folder_path)
        os.mkdir(folder_path)
    if len(glob.glob(folder_path + '/*')) == 0:
        return
    choose = input(folder_path + ' is not empty, \
        choose to remove all content(y), or not remove continue(nc), or exit(e)')
    if choose == 'y':
        files = glob.glob(folder_path + '/*')
        for f in files:
            os.remove(f)
        assert len(glob.glob(folder_path + '/*')) == 0, 'not remove all files ( may be folder ? )'
        return
    elif choose == 'nc':
        return
    else:
        assert False

def data_json_setting_load(path):
    with open(path, 'r') as f:
        read_info = json.load(f)
    print('running', read_info['name'])
    write_json_path = path.replace("0_run", read_info['name'])
    assert os.path.exists(write_json_path), write_json_path + " not exists"
    with open(write_json_path, 'r') as f:
        write_info = json.load(f)
    for key in read_info.keys():
        if key == "result-log":
            continue
        assert read_info[key] == write_info[key], \
                "not match" + str(read_info[key]) + '\n' + str(write_info[key])
    return write_info, write_json_path