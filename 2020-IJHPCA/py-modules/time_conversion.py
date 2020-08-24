#!/usr/bin/env python

import re
from datetime import datetime, timedelta

import sys
if sys.version_info.major < 3:
    from timezones import Pacific, utc
else:
    from .timezones import Pacific, utc

CLI_DATEFORMAT = "%Y-%m-%d"
#SIMCSV_TIMEFORMAT = "%Y-%m-%dT%H-%M-%S"
re_dhms = re.compile(r'^\s*(\d+)[:-](\d+):(\d+):(\d+)\s*$')
re_hms  = re.compile(r'^\s*(\d+):(\d+):(\d+)\s*$')

def datetime_to_iso(dt):
    return dt.isoformat()

def datetime_to_epoch(dt):
    return int((dt - datetime(1970, 1, 1, tzinfo=utc)).total_seconds())

def timedelta_to_walltime_str(tdelta):
    hours, remainder = divmod(tdelta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))

def timedelta_to_seconds_str(tdelta):
    return str(int(tdelta.total_seconds()))

def datestring_to_datetime(datestring, string_format=CLI_DATEFORMAT, tz=Pacific):
    output_time = datetime.strptime(datestring, string_format)
    output_time = output_time.replace(tzinfo=tz)
    return output_time

def walltime_str_to_timedelta(walltime_str):
    (days, hours, mins, secs) = (0, 0, 0, 0)
    match = re_dhms.search(walltime_str)
    if match:
        days  = int(match.group(1))
        hours = int(match.group(2))
        mins  = int(match.group(3))
        secs  = int(match.group(4))
    else:
        match = re_hms.search(walltime_str)
        if match:
            hours = int(match.group(1))
            mins  = int(match.group(2))
            secs  = int(match.group(3))
    return timedelta(days=days, hours=hours, minutes=mins, seconds=secs)

def timestring_to_timedelta(timestring):
    '''
    Convert a string represent # of seconds to timedelta object
    '''
    return timedelta(seconds=int(timestring))

def epoch_to_datetime(epoch, tz=Pacific):
    '''
    Convert a unix epoch to datetime object
    Use PST timezone
    '''
    return datetime.fromtimestamp(epoch, tz=tz)

def epochstring_to_datetime(epochstring, tz=Pacific):
    '''
    Convert a unix epoch string to datetime object
    Use PST timezone
    '''
    return epoch_to_datetime(int(epochstring), tz=tz)

