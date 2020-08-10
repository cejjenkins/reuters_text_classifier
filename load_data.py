from reuters import ReutersSGMLParser

parser = ReutersSGMLParser()
data = parser.empty_row()
for path in  ['data/reuters21578/reut2-000.sgm']:
    # parse current document
    rows = parser.parse(path)
    # append rows into dataset
    for key in data.keys():
        data[key] = data[key] + rows[key]

df = pd.DataFrame(data, columns=data.keys())
#df = df.astype(dtype= {"date":"datetime64[]"})
df.head()