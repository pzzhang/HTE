from VectorFussedLasso import VectorFussedLasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load data
df = pd.read_csv("data/bingads_mm.csv", index_col=0).fillna(0.0)
selectMetrics = {'OverallCoverageOfAdGroupOrEmbeddedSliced'}
df['selected'] = map(lambda x : x in selectMetrics, df['Metric'])
df = pd.DataFrame(df[df['selected'] == True])

# normal processing
metrics = df['Metric'].values[::14]
nmetric = len(metrics)
dates = df['Date'].values[:14]
ndate = len(dates)
# compute the metrics to be filtered out
df['filtered'] = np.logical_or(np.abs(df['AvgC']) < 1e-16, np.abs(df['AvgT']) < 1e-16)
df['filtered'] = np.logical_or(df['filtered'], np.logical_or(df['StdC'] < 1e-16, df['StdT'] < 1e-16))
df['filtered'] = np.logical_or(df['filtered'], np.logical_or(df['NC'] < 1e4, df['NT'] < 1e4))
df['filtered'] = np.logical_or(df['filtered'], df['AvgC']*df['AvgT'] < 0)
mask = np.reshape(df['filtered'].values, (ndate, nmetric), order='F').any(axis=0)
metrics_filted = set(metrics[mask])
df['filtered'] = map(lambda x : x in metrics_filted, df['Metric'])
df_filtered = pd.DataFrame(df[df['filtered'] == False])
# compute the delta, its m, relative delta and its m
metrics = df_filtered['Metric'].values[::14]
nmetric = len(metrics)
dates = df_filtered['Date'].values[:14]
ndate = len(dates)

df_filtered['Delta'] = df_filtered['AvgT'] - df_filtered['AvgC']
df_filtered['DeltaVar'] = df_filtered['StdC']**2/df_filtered['NC'] + df_filtered['StdT']**2/df_filtered['NT']
df_filtered['PerDelta'] = np.log(df_filtered['AvgT']/df_filtered['AvgC'])
df_filtered['PerDeltaVar'] = (df_filtered['StdC']/df_filtered['AvgC'])**2/df_filtered['NC'] + (df_filtered['StdT']/df_filtered['AvgT'])**2/df_filtered['NT']

RDelta = np.reshape(df_filtered['PerDelta'].values, (ndate, nmetric), order='F')
VarRDelta = np.reshape(df_filtered['PerDeltaVar'].values, (ndate, nmetric), order='F')

# plot error bar
x = np.arange(ndate)
yy = RDelta
ey = 1.96*np.sqrt(VarRDelta)

fig, ax = plt.subplots()
plt.errorbar(x, yy, ey, marker='^', ecolor='g', label='Ad Coverage')
plt.legend(loc='best')
plt.xlim([-1, ndate])
#plt.ylim([-1.5, 0.5])
ax.set_xticks(np.arange(ndate), minor=False)
ax.set_xticklabels(dates, minor=False, rotation=45)
plt.ylabel('Delta%')
plt.title("Overall Ad Coverage")
fig.autofmt_xdate()
plt.show()