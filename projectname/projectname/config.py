"""This file is the configuration for data directories and paths."""

home_dir = '/Users/caihao'
original_data_dir = home_dir + '/Dropbox/Insight_Jan2020/data'

raw_data_dir = original_data_dir + '/raw'
processed_data_dir = original_data_dir + '/processed'
cleaned_data_dir = original_data_dir + '/cleaned'
tweets_data_dir = home_dir + '/Dropbox/TEMP/tweets'

classlabel2timelength = {
    0: 'no series a',
    1: '0 to 180 days',
    2: '180 to 360 days',
    3: '360 to 540 days',
    4: '540 to 720 days',
    5: '720 to 900 days',
    6: '900 to 1080 days',
    7: '1080 to 10000 days'
}