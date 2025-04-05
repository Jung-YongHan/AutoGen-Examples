import time


def calculate_elapsed_time(start, end):
    """
    Calculate the elapsed time between two timestamps in days, hours, minutes, and seconds.
    :param start: Start timestamp in seconds since epoch.
    :param end: End timestamp in seconds since epoch.
    :return: A tuple containing elapsed days, hours, minutes, and seconds.
    """

    if start > end:
        raise ValueError("Start time must be less than end time.")
    if start < 0 or end < 0:
        raise ValueError("Timestamps must be non-negative.")
    if start == end:
        return 0, 0, 0, 0
    if start == 0:
        start = time.time()
    if end == 0:
        end = time.time()

    elapsed_time = end - start
    elapsed_day = int(elapsed_time // 86400)
    elapsed_hour = int((elapsed_time % 86400) // 3600)
    elapsed_minute = int((elapsed_time % 86400) % 3600 // 60)
    elapsed_second = int((elapsed_time % 86400) % 3600 % 60)
    return elapsed_day, elapsed_hour, elapsed_minute, elapsed_second
