from tqdm import tqdm
import requests
import os

data_set = os.path.join('yoga_pose','82_data')

missing_data = os.path.join('yoga_pose','82_data','missing')

path = os.path.join('yoga_pose','Yoga-82','yoga_dataset_links')

dirs= os.listdir(path)

def get_image(category):
    missing = []
    data_path = os.path.join(path,category)
    with open(data_path,'r') as f:
        for line in tqdm(f):
            split = line.split(',')
            im_path = os.path.join(data_set,split[0])
            try:
                
                response = requests.get(split[1],timeout=10)
            
                if response.status_code == 200:
                    os.makedirs(os.path.dirname(im_path), exist_ok=True)
                    with open(im_path,'wb') as g:
                        g.write(response.content)
                else:
                    #print(line)
                    missing.append(line)
            except:
                #print('\nnot availble\n')
                missing.append(line)
    miss_file = os.path.join(missing_data,'{}.txt'.format(category))
    os.makedirs(os.path.dirname(miss_file), exist_ok=True)
    with open(miss_file,'w') as f:
        f.writelines('%s\n'% miss for miss in missing)

if __name__ == "__main__":
    done=os.listdir(data_set)
    for x in dirs:
        if os.path.splitext(x)[0] not in done and 'desktop' not in x:
            print('getting images form {}'.format(x))
            get_image(x)
            print('Done\n')
    