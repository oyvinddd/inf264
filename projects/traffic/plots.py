import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np

# helper function for visualizing car volume with respect to the selected features
def plot_data(X, y):
    colors = ['r', 'g', 'b']
    for col in range(X.shape[1]):
        plt.figure(1, figsize=(24, 16))
        if col < X.shape[1] - 1:
            plot_idx = col+1
        else:
            plot_idx = 4
        plt.subplot(5, 3, plot_idx)
        plt.scatter(X.iloc[:, col], y.iloc[:, 2], marker='o', c=colors[col])
        plt.xlabel(X.columns[col])
        plt.ylabel('Total volume')
    plt.suptitle("Total car volume with respect to each of the features")
    plt.show()

def plot_weekdays(data):
    total = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }
    dnp = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }
    sntr = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }
    no_of_days = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0
    }
    for index, row in data.iterrows():
        year, month, day, hour = row['År'], row['Måned'], row['Dag'], row['Fra_time']
        date = dt.datetime(year, month, day, hour=hour)
        v_totalt = row['Volum totalt']
        v_dnp = row['Volum til DNP']
        v_sntr = row['Volum til SNTR']
        total[date.weekday()] = total[date.weekday()] + v_totalt
        dnp[date.weekday()] = dnp[date.weekday()] + v_dnp
        sntr[date.weekday()] = sntr[date.weekday()] + v_sntr
        no_of_days[date.weekday()] = no_of_days[date.weekday()] + 1
    avg = {
        0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []
    }
    for k, v in total.items():
        avg[k].append(v / no_of_days[k])
        avg[k].append(sntr[k] / no_of_days[k])
        avg[k].append(dnp[k] / no_of_days[k])

    # create bar chart
    labels = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    tot_cars, sntr_cars, dnp_cars = [], [], []
    for k, v in avg.items():
        tot_cars.append(v[0])
        sntr_cars.append(v[1])
        dnp_cars.append(v[2])
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x + width, dnp_cars, width, label='DNP')
    rects2 = ax.bar(x, sntr_cars, width, label='SNTR')
    rects3 = ax.bar(x - width, tot_cars, width, label='Total')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg. volumes per hour')
    ax.set_title('Avg. car volumes with respect to each weekday')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()

def plot_day(date, data):

    day = data[data.År == date.year]
    day = day[day.Måned == date.month]
    day = day[day.Dag == date.day]
    plt.plot(day.Fra_time, day.Volum_totalt)
    # y_sntr = data.loc[:, 'Volum til SNTR']
    # y_dnp = y.loc[:, 'Volum til DNP']
    # y_total = y.loc[:, 'Volum totalt']
    # plt.plot(x_hour, y_total)
    # plt.plot(x_hour, y_sntr)
    # plt.plot(x_hour, y_dnp)
    # plt.title('24 hours')
    # plt.xlabel('Tid')
    # plt.ylabel('Volum')
    # plt.legend(['Total', 'SNTR', 'DNP'])
    plt.ylabel('Volume')
    plt.xlabel('Time of day')
    plt.title('December 25th, 2019')
    plt.show()

def volume_by_month(year, data):
    year = data[data.År == year]
    months = {}
    for index, row in year.iterrows():
        year, month, day, hour = row['År'], row['Måned'], row['Dag'], row['Fra_time']
        date = dt.datetime(year, month, day, hour=hour)
        volum_totalt = row['Volum totalt']
        if month not in months:
            months[month] = volum_totalt
        else:
            months[month] += volum_totalt
    return months


# plot months:

data = pd.read_csv('data.csv')
# plot_day(dt.datetime(2019, 1, 25), data)
m16 = volume_by_month(2016, data).values()
m17 = volume_by_month(2017, data).values()
m18 = volume_by_month(2018, data).values()
m19 = volume_by_month(2019, data).values()

plt.xlabel('Months')
x_labels = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
plt.plot(x_labels, m16, color='red')
plt.plot(x_labels, m17, color='blue')
plt.plot(x_labels, m18, color='green')
plt.plot(x_labels, m19, color='pink')
plt.show()
