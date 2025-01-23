import pingouin as pg
import pandas as pd
import numpy as np


#Hand Clocks
data_handclocks_general = pd.DataFrame({
    'subject': np.tile(np.arange(1,28),3),
    'judge': np.repeat([1,2,3],27),
    'score': [0,3,1,0,0,3,1,3,4,4,3,3,4,3,3,4,4,4,3,4,4,3,4,3,4,3,2,0,1,0,0,0,0,1,2,4,4,3,3,3,3,3,4,4,3,2,3,3,3,3,3,3,3,1,0,1,1,1,1,1,2,1,4,4,3,2,2,4,4,2,4,1,1,1,2,3,4,3,3,3,1]
})

#Data Numbers
data_numbers_general = pd.DataFrame({
    'subject': data_length,
    'JuanCarlos': [0,3,0,0,1,3,2,2,4,4,4,4,4,2,4,4,2,4,4,2,4,2,4,4,4,4,2],
    'Mauricio': [0,3,0,0,2,2,1,3,4,4,4,3,4,4,3,3,3,3,3,3,3,3,3,4,4,3,2],
    'MoCASystem': [0,2,0,0,2,4,2,0,4,4,4,3,4,2,2,4,3,3,4,4,3,2,4,4,3,3,1]
})



# Compute ICC
print("HandClock Evaluating General")
icc = pg.intraclass_corr(data=data_handclocks_general, targets='subject', raters='judge', ratings='score')
print(icc)
