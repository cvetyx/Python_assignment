#!/usr/bin/env python
# coding: utf-8

import requests
import json
import pandas as pd
import numpy as np

url = 'https://api.ycombinator.com/companies/export.json?callback=true'

resp = requests.get(url=url)
data = resp.text[15:-2]
y = json.loads(data)
for el in y:
    print(el['name'], el['batch'], el['description'], el['url'])

df = pd.DataFrame(y)
print(df)

df.to_csv(r"C:\Users\Lina\Desktop\scrapped-comapnies.csv")




