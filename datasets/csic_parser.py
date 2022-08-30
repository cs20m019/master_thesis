'''
https://github.com/Monkey-D-Groot/Machine-Learning-on-CSIC-2010
'''

import os
import sys
import urllib.parse
import io

normal_train_raw = 'normalTrafficTraining.txt'
normal_test_raw = 'normalTrafficTest.txt'
anomaly_file_raw = 'anomalousTrafficTest.txt'

normal_train_file_parse = 'normal_train_request.txt'
normal_test_file_parse = 'normal_test_request.txt'
anomaly_file_parse = 'anomalous_test_request.txt'

'''
Function parses files from HTTP dataset CSIC2010 and writes them to files
'''
def parse_file(file_in, file_out):
    fin = open(os.path.join(sys.path[0], file_in), "r")
    fout = io.open(file_out, "w", encoding="utf-8")
    lines = fin.readlines()
    res = []
    for i in range(len(lines)):
        line = lines[i].strip()
        # HTTP GET request is concatented by http-method + url
        if line.startswith("GET"):
            res.append("GET" + line.split(" ")[1])
        # HTTP POST is concatened by http-method + url, the url is determined by the Content-Length that is posted
        elif line.startswith("POST") or line.startswith("PUT"):
            url = line.split(' ')[0] + line.split(' ')[1]
            j = 1
            while True:
                if lines[i + j].startswith("Content-Length"):
                    break
                j += 1
            j += 1
            data = lines[i + j + 1].strip()
            url += '?' + data
            res.append(url)
    for line in res:
        line = urllib.parse.unquote(line).replace('\n','').lower()
        fout.writelines(line + '\n')
    print ("finished parse ",len(res)," requests")
    fout.close()
    fin.close()

# Parse all files HTTP CSIC2010 files
if not os.path.exists('normal_train_request.txt') or not os.path.exists('normal_test_request.txt') or not os.path.exists('anonalous_test_request.txt'):
    parse_file(normal_train_raw,normal_train_file_parse)
    parse_file(normal_test_raw,normal_test_file_parse)
    parse_file(anomaly_file_raw,anomaly_file_parse)