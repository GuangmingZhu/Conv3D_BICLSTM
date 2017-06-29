import os
import io
import sys
import numpy as np

check_datalist = ('train_rgb_list.txt','train_depth_list.txt','train_flow_list.txt',
		'valid_rgb_list.txt','valid_depth_list.txt','valid_flow_list.txt',
		'test_rgb_list.txt','test_depth_list.txt','test_flow_list.txt')
for fileid in xrange(len(check_datalist)):
  print 'Checking the completeness of the images indicated by %s' % check_datalist[fileid]
  f = open(check_datalist[fileid], 'r')
  f_lines = f.readlines()
  f.close()
  for idx, line in enumerate(f_lines):
    videopath  = line.split(' ')[0]
    framecnt   = int(line.split(' ')[1])-1
    for idx in xrange(1, framecnt+1):
      image_file = '%s%06d.jpg' %(videopath, idx)
      try:
        assert os.path.exists(image_file)
      except:
        print '%s does not exist' % image_file
