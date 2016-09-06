from VectorFussedLasso import VectorFussedLasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load data
df = pd.read_csv("data/bingads_mm.csv", index_col=0).fillna(0.0)
# if the metric does not have

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

# creat an object
chambo2 = VectorFussedLasso(RDelta, 1.0/VarRDelta, dates)
print "Initial Noise_Energy", chambo2.c0/2
# print "Initial Constant Term", chambo2.c0

# en, U = chambo2.P_PDALG1(Lambda = 1e-4, n_it = 30000)
# flux = chambo2.get_flux(U)

# print the very first step and the original data
chambo2.print_effect(u=chambo2.U0, lam=0, stepid=-1, str_method="LassoPath", xlabel=dates)
chambo2.print_effect(u=RDelta, lam=1e16, stepid=1e16, str_method="LassoPath", xlabel=dates)

# test pure lasso
u_lasso_path, noise_lasso_path, uls_lasso_path = chambo2.lasso_path(lam_min=1e-2, lam_max=1e-1, nstep=400)
