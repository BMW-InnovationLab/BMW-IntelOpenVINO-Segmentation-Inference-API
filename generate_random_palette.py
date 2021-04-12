import sys, getopt
import shutil
import os
import json


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0, n):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while (lab > 0):
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete





def get_configuration(model_name):
    base_models_dir = './model'
    try:
        model_path = os.path.join(base_models_dir, model_name)
    except Exception as ex:
        raise ex

    path = os.path.join(model_path,'configuration.json')

    try:
        with open(path) as f:
            configuration = json.loads(f.read())		
    except Exception as ex:
        raise ex
    return configuration["classes"],model_path

    

def main(argv):

    try:
        if not argv or argv[0] == '-h':
            print 'generate_random_palette.py -m <ModelName>'
            sys.exit()
        elif argv[0] in ("-m", "--model"):

            class_num,model_path = get_configuration(argv[1])

            palette= _getvocpallete(class_num)

            full_path = model_path + '/palette.txt'
            with open(full_path,'w') as filehandle:
                for i in palette:
                    filehandle.write('%s\n' % i)
                filehandle.close()
            print("Done!")

    except getopt.GetoptError:
        print 'generate_random_palette.py -m <ModelName>'
        sys.exit(2)

if __name__ == "__main__":
    
    main(sys.argv[1:])
