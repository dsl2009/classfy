import csv
import glob
def get_labels_map():
    d = dict()
    for x in open('label.txt',encoding='utf8').readlines():
        data = x.replace('\n','').split('\t')
        d[data[0]] =data[1]
    return d

def gen_pred():
    lb_idx = get_labels_map()
    l1 = glob.glob('D:/deep_learn_data/luntai/pred/other/*/*.jpg')
    with open('perd07.csv', "w", newline='') as f:
        writer = csv.writer(f)
        for x in l1:
            print(x)
            data = x.replace('\\','/').split('/')
            image_id = data[-1]
            label = data[-2]
            print(image_id, label)
            print(image_id, label, lb_idx[label])
            writer.writerow([image_id, lb_idx[label]])
            f.flush()





if __name__ == '__main__':
    gen_pred()