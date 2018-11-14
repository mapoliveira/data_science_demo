import pandas as pd

fert = pd.read_csv('../rawData/gapminder_total_fertility.csv', index_col=0)
life = pd.read_excel('../rawData/gapminder_lifeexpectancy.xlsx', index_col=0)
pop = pd.read_excel('../rawData/gapminder_population.xlsx', index_col=0)

printInfo = 0

if printInfo ==1: 
   print('\n########## Life expectancy info: ##########')
   
   print('\n--> Life expectancy summary:') 
   # rows=countries, columns=1800-2016 
   print(life.describe())
   print('\n--> Life expectancy raw data size: '+ str(life.shape) +'\n')
   print(life.head(5))
   
   print('\n########## Fertility info: ##########') 
   # rows=countries, columns=1800-2015
   print('\n--> Fertility summary:')
   print(fert.describe())
   print('\n--> Fertility raw data size: '+ str(fert.shape) +'\n')
   print(fert.head(5))
 
   print('\n########## Population info: ##########') 
   # rows=countries, columns=1800-2015
   print('\n--> Population summary:')
   print(pop.describe())
   print('\n--> Population raw data size: '+ str(pop.shape) +'\n')
   print(pop.head(5))

# Merge data considering that all dataframes have years in columns:
ncol = [int(x) for x in fert.columns]
fert.set_axis(axis=1, labels=ncol, inplace=True)

ncol = [int(x) for x in life.columns] 
life.set_axis(axis=1, labels=ncol, inplace=True)

print(pop.head(5))
ncol = [int(x) for x in pop.columns]
pop.set_axis(axis=1, labels=ncol, inplace=True)
print(pop.head(5))

sfert = fert.stack()
slife = life.stack()
spop = pop.stack()

d = {'fertility': sfert, 'lifeexp': slife, 'population': spop}
df2 = pd.DataFrame(data=d)
print(df2.head(10))

df3 = df2.stack()
print(df2.head(10))

df4 = df3.unstack((0,2))
print(df2.head(10))


import pylab as plt
df4[['Germany', 'France', 'Sweden']].plot()

df5 = df3.unstack(2)
df5.plot.scatter('fertility', 'lifeexp', s=0.1)

df6 = df3.unstack(1)
df6 = df6[1950]
df6 = df6.unstack(1)
df6.plot.scatter('fertility', 'lifeexp', s=0.1)

cmap = plt.get_cmap('tab20').colors
df6.plot.scatter('fertility', 'lifeexp', s=0.1, c=cmap)

df6.plot.scatter('fertility', 'lifeexp', s=df6['population'])

import imageio

images = []

for i in range(0, 100):
    filename = 'lifeexp_{}.png'.format(i)
    images.append(imageio.imread(filename))

imageio.mimsave('output.gif', images, fps=20)

