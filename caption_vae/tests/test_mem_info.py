# -*- coding: utf-8 -*-
"""
Created on 06 Oct 2020 23:03:40
@author: jiahuei
"""
import timeit

test_mem = ''' 
def get_memory_info():
    """
    Get node total memory and memory usage
    https://stackoverflow.com/a/17718729
    """
    with open("/proc/meminfo", "r") as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == "MemTotal:":
                ret["total"] = int(sline[1])
            elif str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                tmp += int(sline[1])
        ret["free"] = tmp
        ret["used"] = int(ret["total"]) - int(ret["free"])
    return ret
'''

print(timeit.repeat(stmt=test_mem))

# Only took [0.07946336700001666, 0.06230000300001848, 0.052760936000140646, 0.04949938999993719, 0.04950043500002721]
# seconds to run 1 million times
