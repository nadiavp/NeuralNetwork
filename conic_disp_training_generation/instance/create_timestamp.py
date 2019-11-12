def create_timestamp(year,month,day,hour=0,min=0,length=1,dt=1):

    from datetime import (datetime,timedelta)

    start = datetime(year,month,day,hour,min,0)
    timestamp = [start]

    for add in range(1,length):
        next = start + timedelta(hours=add*dt)
        timestamp.append(next)

    return timestamp