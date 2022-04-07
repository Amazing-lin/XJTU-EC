import pickle
import numpy as np
import pandas as pd
# pickle.dump(data,f)
m = open('/home/bell-chuangxingang/桌面/LIu/out_table/data_0.pickle','rb')
inf = pickle.load(m)
# print(inf)
# inf = bytes(inf,encoding='utf8')
# f = open('/home/bell-chuangxingang/桌面/LIu/out_table/data0.txt','wb')
# f.write(inf)
# # for a in range(len(inf)):
# for b in inf[0]:
infma = np.array(inf[0])
# np.savetxt('/home/bell-chuangxingang/桌面/LIu/out_table/data0.txt',infma)
data = pd.DataFrame(infma)
data.to_csv('/home/bell-chuangxingang/桌面/LIu/out_table/data0.csv')

