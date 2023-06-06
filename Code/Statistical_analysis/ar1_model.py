import pandas as pd
import numpy as np
import pymc as pm
import pytensor
import pytensor.printing as tp
import pytensor.tensor as tt
import pymc.sampling.jax as pyjax #- need cuda 11.8 and above for that
pytensor.config.exception_verbosity = 'high'

import arviz as az
az.rcParams["plot.matplotlib.show"] = True  # bokeh plots are automatically shown by default

# Read the data
data_thermo_95 = pd.read_csv("Data/data_for_statistical_analysis.csv")

# Convert factors to representing numbers
wrong = data_thermo_95["wrong_sun.shade"].values
part_of_day = data_thermo_95["part_of_day"].factorize(sort=True)
true_color = data_thermo_95["true_color"].factorize(sort=True)
true_thermo = data_thermo_95["true_thermo"].factorize(sort=True)
enclosure, _ = data_thermo_95["enclosure"].factorize(sort=True)
data_thermo_95["camera_side"] = data_thermo_95["camera"].str[2]
camera, _ = data_thermo_95["camera"].factorize(sort=True)

#convert decimal time to seconds
h = data_thermo_95["h"].values * 60 * 60

# Create the X matrix for the model
#main effects
morning = (part_of_day[0] == 1).astype(int)
noon = (part_of_day[0] == 2).astype(int)
red = (true_color[0] == 1).astype(int)
white = (true_color[0] == 2).astype(int)
sun = (true_thermo[0] == 1).astype(int)
# 1-way interactions
morning_red = morning * red
morning_white = morning * white
noon_red = noon * red
noon_white = noon * white
morning_sun = morning * sun
noon_sun = noon * sun
red_sun = red * sun
white_sun = white * sun
# 2-way interactions
morning_red_sun = morning_red * sun
morning_white_sun = morning_white * sun
noon_red_sun = noon_red * sun
noon_white_sun = noon_white * sun

#create the matrix
X = np.column_stack([np.ones_like(morning), morning, noon, red, white, sun,
                     morning_red, morning_white, noon_red, noon_white,
                     morning_sun, noon_sun, red_sun, white_sun,
                     morning_red_sun, morning_white_sun, noon_red_sun, noon_white_sun])

# Number of samples
N = X.shape[0]

# Compute time differences in h values for the ar1
h_diff = np.diff(h)
h_diff = np.concatenate([np.array([0]), h_diff])  # Add a zero at the beginning
h_diff = np.clip(h_diff, -100, 100)
y_mean = tt.zeros_like(h)
same_camera_mask = camera[1:] == camera[:-1]
same_camera_mask = np.concatenate([np.array([0]), same_camera_mask])
cameras = np.array(range(10))

# Now use the variables in the PyMC3 code
with pm.Model() as model:
    #sigmbeta = pm.HalfNormal("sigmbeta", sigma=10)
    # Fixed effects
    beta = pm.Normal("beta", mu=0, sigma=2, shape=18)

    # Random effects
    sigmenc = pm.HalfNormal("sigmenc", sigma=2)
    sigmcam = pm.HalfNormal("sigmcam", sigma=2)
    enc = pm.Normal("enc", mu=0, sigma=sigmenc, shape=5)
    cam = pm.Normal("cam", mu=enc[cameras//2], sigma=sigmcam, shape=10)

    # AR-1 parameters
    rho = pm.HalfNormal("rho", sigma=2, shape=10)
    decay_coef = pm.Uniform("decay_coef", lower = 0.000, upper = 1, shape=10)
    decay_par = rho[camera]*(decay_coef[camera]**h_diff)*same_camera_mask

    # Compute the linear predictor without ar-1
    yp0 = pm.math.dot(X,beta)+ cam[camera]
    p = pm.math.invlogit(yp0) #pm.mBernoulli.dist(logit_p=yp0)
    
    # calculate ar-1 - needed two steps to get rid of temporal autocorrelation
    res = wrong[:-1]-p[:-1]
    yp0_p = yp0[1:] + res*decay_par[1:]
    p = pm.math.invlogit(yp0_p)
    res = wrong[:-2] - p[:-1]
    yp0_p = yp0_p[1:] + res*decay_par[2:]
    
    #save predictions in p_ar1
    p_ar1 = pm.Deterministic("p_ar1", pm.math.invlogit(yp0_p))
    # Bernoulli likelihood
    obs = pm.Bernoulli("obs", logit_p=yp0_p, observed=wrong[2:])


    # Sample using nuts
    trace = pyjax.sample_numpyro_nuts(1000, tune=1000, chains=3, idata_kwargs={"log_likelihood": True})


#check model convergence using traceplots
az.plot_trace(trace, var_names=('beta'))
#get model estimates and Rhats for parameters
summary = az.summary(trace, var_names=["beta", "rho"])
print(summary)

#create posterior predictions for model evaluation
with model:
    pp_samples = pm.sample_posterior_predictive(trace, var_names=["obs", "p_ar1"], extend_inferencedata=True)

#save model traces 
trace.to_netcdf("trace_ar1.nc")
pp_samples.to_netcdf("pp_ar1.nc")

#convert traces to dictionaries for further analysis
trace_samples = trace.to_dict()["posterior"]
pp_samples = pp_samples.to_dict()["posterior_predictive"]

### checking bayesian p-value
#get the data for a single chain (the second chain in this example)
p_mean_samples = pp_samples["p_ar1"][1]
p_obs_samples = pp_samples["obs"][1]
#calculate the mean replicated data for each data point
p_obs_samples_mean = np.mean(p_obs_samples, axis=(0))

# Compute the chi-squared values for replicated data using vectorization
chisq_r_samples = np.sum(
    np.square(
        (p_obs_samples - p_mean_samples) / np.maximum(np.sqrt(p_mean_samples * (1 - p_mean_samples)), 0.001)
    ),
    axis=1
)

# Compute the chi-squared values for observation data using vectorization
data_thermo_95 = pd.read_csv("data_for_statistical_analysis.csv")
wrong = data_thermo_95["wrong_sun.shade"].values
chisq_o_samples = np.sum(
    np.square(
        (wrong[2:] - p_mean_samples) / np.maximum(np.sqrt(p_mean_samples * (1 - p_mean_samples)), 0.001)
    ),
    axis=1
)

#calculate the p-value
chisq_diff = chisq_r_samples - chisq_o_samples
chisq_p = np.mean(chisq_diff > 0)
print("chisq_p:", chisq_p)

#### Check residuals autocorrelation
#load relevant plotting library
import matplotlib.pyplot as plt 

#Compute the Pearson residuals
#mean model predictions
p_mean = np.mean(pp_samples["p_ar1"], axis=(0, 1))
#residual calculations
pearson_residuals = (wrong[2:] - p_mean) / np.sqrt(p_mean * (1 - p_mean))

# Plot the autocorrelation 
plt.acorr(pearson_residuals, maxlags=20, usevlines=True, normed=True, lw=1)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation of Pearson Residuals")
plt.show()

#more post predictive checks
az.plot_ppc(trace)
