import os

def convert(source_path, target_path):
    '''
    with open(source_path) as f:
        text = f.read()
    name_list = text.split('\n')[:-1]
    '''
    name_list = []
    for name_with_ext in os.listdir(source_path):
        name,ext = os.path.splitext(name_with_ext)
        name_list.append(name)

    root = os.path.dirname(os.path.realpath(__file__))
    target_name_list = []
    for name in name_list:
        target_name = os.path.join(root,'data',name) + '.jpg'
        target_name_list.append(target_name)

    with open(target_path, 'w') as f:
        f.write('\n'.join(target_name_list))
        f.write('\n')

convert('annotation_YOLO/', 'index_YOLO/trainval.txt')
