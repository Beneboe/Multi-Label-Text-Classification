# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# %%
df = pd.read_csv('datasets/wiki_stats.csv')

# %%
years = mdates.YearLocator(base=2)
months = mdates.MonthLocator()
years_fmt = mdates.DateFormatter('%Y')

# %%
dates = df['month'].to_numpy().astype(np.datetime64)
articles = df['total.content']

fig, ax = plt.subplots()

# plot
ax.plot(dates, articles, linewidth=2)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(dates[0], 'Y')
datemax = np.datetime64(dates[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# y axis
# ax.ticklabel_format(axis='y', style='plain')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax.set_title('Number of English Wikipedia Articles')
ax.set_xlabel('Year')
ax.set_ylabel('Articles')

ax.grid(axis='y')

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

fig.tight_layout()
fig.savefig(f'datasets/wiki_articles.png', dpi=163)
fig.savefig(f'datasets/wiki_articles.pdf')

# %%
