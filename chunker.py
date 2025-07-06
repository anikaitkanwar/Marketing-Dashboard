from datetime import datetime, timedelta

def split_date_range(start, end, parts=7):
    total_duration = end - start
    delta = total_duration / parts
    ranges = []

    for i in range(parts):
        part_start = start + i * delta
        part_end = start + (i + 1) * delta
        ranges.append((part_start, part_end))
    
    return ranges
