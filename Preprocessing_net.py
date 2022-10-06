
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps

# Data
data_deaths = pd.read_csv("D:\\Universiteit\\Technische Wiskunde\\AM2050 - Modelling 2A\\supsmu.csv")
data_deaths = pd.DataFrame(data_deaths)
deaths_smooth = data_deaths.iloc[:, [data_deaths.columns.get_loc("supsmu_D.y")]]
deaths_smooth = np.array(deaths_smooth["supsmu_D.y"])
deaths = data_deaths.iloc[:, [data_deaths.columns.get_loc("deaths")]]
deaths = np.array(deaths["deaths"])
dates = data_deaths.iloc[:, [data_deaths.columns.get_loc("dates")]]
dates = np.array(dates["dates"])

mean_IP = 5.6
median_IP = 5
mean_IOD = 14.5
median_IOD = 13.2
mean_IP_IOD = 20.1
median_IP_IOD = 18.8


# Utilized Functions
def mu_sigma(mean, median):
    mu = np.log(median)
    sigma = np.sqrt(2*np.log(mean/median))
    return mu, sigma

def lognormal_IP(x):
    u, o = mu_sigma(mean_IP, median_IP)
    log = np.array([])
    for i in x:
        if i > 0:
            log = np.append(log, np.exp(-(np.log(i)-u)**2/(2*o**2))/(i*o*np.sqrt(2*np.pi)))
        else:
            log = np.append(log, 0)
    return log

def lognormal_IOD(x):
    u, o = mu_sigma(mean_IOD, median_IOD)
    log = np.array([])
    for i in x:
        if i > 0:
            log = np.append(log, np.exp(-(np.log(i)-u)**2/(2*o**2))/(i*o*np.sqrt(2*np.pi)))
        else:
            log = np.append(log, 0)
    return log

def lognormal_IP_IOD(x):
    u, o = mu_sigma(mean_IP_IOD, median_IP_IOD)
    log = np.array([])
    for i in x:
        if i > 0:
            log = np.append(log, np.exp(-(np.log(i)-u)**2/(2*o**2))/(i*o*np.sqrt(2*np.pi)))
        else:
            log = np.append(log, 0)
    return log

def convolution(x, n, f1, f2):
    conv = np.array([])
    for xx in x:
        xp = np.linspace(0,xx,n)
        h = f1(xp)*f2(xx-xp)
        I = simps(h, xp)
        conv = np.append(conv, I)
    return conv

def F_disc_log(n):
    n_array = np.linspace(n, n+1, 50)
    return simps(lognormal_IP_IOD(n_array), n_array)

def y(n_0, x, F):
    conv = np.array([])
    for n0 in n_0:
        print(n0)
        sum_ = []
        for n in np.arange(n0,len(x)):
            sum_.append(x[n]*F(n-n0))
        I = sum(sum_)
        conv = np.append(conv, I)
    return conv

def total_deaths(start_index, end_index):
    return sum(deaths[start_index:end_index])

def cfr_2(prcnt_diff_inf, start_date, end_date):
    population_size = 17450000
    
    new_inf = prcnt_diff_inf*population_size
    
    start_index = np.where(dates == start_date)[0][0]
    end_index = np.where(dates == end_date)[0][0]
    
    cfr_value = total_deaths(start_index, end_index)/new_inf
    
    return cfr_value

def cfr_sero(prcnt_inf_start_end_array):
    date_cfr_array = []
    
    for i in range(len(prcnt_inf_start_end_array)):
        date_cfr_array.append(cfr_2(prcnt_inf_start_end_array[i][0], prcnt_inf_start_end_array[i][1], prcnt_inf_start_end_array[i][2]))
    
    return date_cfr_array

def cfr_array_2():
    new_inf_date_array = [[0.005, "01/07/2020", "01/10/2020"], [0.07, "01/10/2020", "01/03/2021"], [0.08, "01/03/2021", "01/07/2021"]]
    october = np.where(dates == "01/10/2020")[0][0]
    maart = np.where(dates == "01/03/2021")[0][0]
    juli = np.where(dates == "01/07/2021")[0][0]
    cfrs = cfr_sero(new_inf_date_array)
    
    days = []
    cfr = []
    
    for n in range(october):
        days.append(n)
        cfr.append(cfrs[0])
    
    for n in range(october, maart):
        days.append(n)
        cfr.append(cfrs[1])
    
    for n in range(maart, juli):
        days.append(n)
        cfr.append(cfrs[2])
        
    return days, cfr

def cfr_array_5():
    new_inf_date_array = [[0.005, "01/07/2020", "01/10/2020"], [0.07, "01/10/2020", "01/03/2021"], [0.08, "01/03/2021", "01/07/2021"]]
    sero_cfr = cfr_sero(new_inf_date_array)
    b = (-2272/17243)*sero_cfr[0] + (21442/17243)*sero_cfr[1] + (-1927/17243)*sero_cfr[2]
    c = (1763*sero_cfr[2] - 130*b)/1633
    a = sero_cfr[0] 
    
    days = np.arange(366)
    
    cfr = []
    
    for n in days:
        if n <= 92:
            cfr.append(a)
        elif 92 < n <= 123:
            cfr.append((b-a)*n/31 + 123*a/31 - 92*b/31)
        elif 123 < n <= 196:
            cfr.append(b)
        elif 196 < n <= 282:
            cfr.append((c-b)*n/86 + 282*b/86 - 196*c/86)
        elif 282 < n:
            cfr.append(c)
            
    return days, np.array(cfr)

def new_infected_5():
    days, cfr = cfr_array_5()
    y_values = y(days, deaths, F_disc_log)
    new_inf = []
    for day in days:
        new_inf.append((y_values[day]/cfr[day]))
    
    return days, new_inf

def total_infected_per_day_5():
    days, new_inf = new_infected_5()
    tot_inf = []
    for i in range(13,len(days)):
        tot_inf.append(sum(new_inf[i-13:i+1]))
    return days[13:], tot_inf

def data_two(start, end):
    data2 = pd.read_json('D:\\Universiteit\\Technische Wiskunde\\AM2050 - Modelling 2A\\data2.json')
    data2 = pd.DataFrame(data2)
    date_start = data2.index[data2["Date"] == start][0]
    date_end = data2.index[data2["Date"] == end][0]
    data2 = data2[date_start:date_end]
    data2 = data2[["Date", "prev_low", "prev_avg", "prev_up"]]
    data2 = data2[::1].reset_index(drop=True)
    return data2

def prev_plot(data2):
    t = np.linspace(0, len(data2["prev_low"]) - 1, len(data2["prev_low"]))
    plt.plot(data2["prev_low"], c='w', alpha=0)
    plt.plot(data2["prev_avg"], label="Estimation RIVM")
    plt.plot(data2["prev_up"], c='w', alpha=0)
    plt.fill_between(t, data2["prev_low"], data2["prev_up"], color='orange')
    plt.grid()

# PLOTS

# PDF PLOT
x = np.linspace(0, 50, 1000)

plt.plot(x, lognormal_IP(x), label="$g(t)$: Incubation period, lognormal, mean=5.6, median=5")
plt.plot(x, lognormal_IOD(x), label="$h(t)$: Illness onset to death, lognormal, mean=14.5, median=13.2")
plt.plot(x, convolution(x, 100, lognormal_IP, lognormal_IOD), label="$f(t)$:Infection to death")
plt.plot(x, lognormal_IP_IOD(x), "--", label="Lognormal, mean=20.1, median=18.8")
plt.legend(fontsize=7.5)
plt.grid()
plt.xlabel("X (days)")
plt.show()

# y[n] PLOT
days = np.arange(0, len(deaths))

plt.plot(days, deaths, label="Number of Covid deaths")
plt.plot(days, y(days, deaths, F_disc_log), label="$y[n]$")
plt.legend()
plt.grid()
plt.xlabel("t (days)")
plt.show()

# CFR PLOT
#days_cfr1, cfr_n1 = cfr_array_2()
days_cfr2, cfr_n2 = cfr_array_5()

#plt.plot(days_cfr1, np.array(cfr_n1)*100, ".")
plt.plot(days_cfr2, np.array(cfr_n2)*100)
plt.grid()
plt.xlabel("Number of Days since 01/07/2020")
plt.ylabel("IFR (%)")
plt.show()

# total infected PLOT
data2 = data_two("2020-08-23 00:00:00", "2021-06-09 00:00:00")

dagen, infected_tot = total_infected_per_day_5()

starting = np.where(dates == "23/08/2020")[0][0]
ending = np.where(dates == "09/06/2021")[0][0]

dagen_nieuw = np.array(dagen[starting-13:ending-13])-53
infected_tot_nieuw = np.array(infected_tot[starting-13:ending-13])

prev_plot(data2)
plt.plot(dagen_nieuw, infected_tot_nieuw, color="cyan", label="Estimation REMEDID")
plt.legend()
plt.xlabel("n (days)")
plt.ylabel("Total Current Infected")
plt.show()