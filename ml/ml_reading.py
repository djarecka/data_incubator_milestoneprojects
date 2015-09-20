import ujson
import pandas as pd

def reading_df():
    f = open('yelp_train_academic_dataset_business.json')
    
    variables_req = ["city", "latitude", "longitude", "stars", "categories", "attributes"]
    df = pd.DataFrame(columns = variables_req)

    for i, line in enumerate(f.readlines()):
        data = ujson.loads(line)
        df.loc[i]=[data[var] for var in variables_req]
        print i/37938.

    df.to_csv("yelp_reqcol.csv")

#reads original file and create file with list of dictionary (as data.py)
def reading_listdict():
    f = open('yelp_train_academic_dataset_business.json')
    data_list = []
    for i, line in enumerate(f.readlines()):
        data_list.append(ujson.loads(line))

    f_out = open("list_data.txt", "w")
    json.dump(data_list, f_out)
    f_out.close()
            
    
