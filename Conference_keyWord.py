#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import time, ipdb, re
import nltk
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

# Set up a browser to crawl from dynamic web pages 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
nltk.download('stopwords')

conference_re_map = {"CVPR":"//dt[@class='ptitle']/a",
					"ECCV":"//dt[@class='ptitle']/a",
					"ICCV":"//dt[@class='ptitle']/a",
					"ICLR":"//li[@class='note ']/h4",
					"ICML":"//div[@class='col-xs-9']/p/b",
					"icml":"//p[@class='title']",
					"NIPS":"//ul/li/a[starts-with(@href, '/paper')]"}

'''to find most frequently appeared words'''
def keyWordVis(http_list, conf, stopwords_deep_learning, num_keyword=0):
	chrome_options = webdriver.ChromeOptions()
	chrome_options.add_argument('--headless')
	chrome_options.add_argument('--no-sandbox')
	chrome_options.add_argument('--disable-dev-shm-usage')
	wd = webdriver.Chrome(ChromeDriverManager().install())

	html_link = []
	title = []

	for http_urls in http_list:
	    html_link = []
	    while len(html_link) == 0:
	        wd.get(http_urls) #FIXME
	        html_link = wd.find_elements_by_xpath(conference_re_map[conf])
	        time.sleep(10)
	    title.extend([hi.text for hi in html_link])
	wd.quit()

	print("The number of total accepted paper titles : ", len(title))
	keyword_list = []

	for i, link in enumerate(title):
	  word_list = title[i].split(" ")
	  add_word = re.split('[\?\- ]', title[i])
	  word_list.extend(list(set(add_word) - set(word_list)))
	  # word_list = list(set(word_list))
	    
	  word_list_cleaned = [] 
	  for word in word_list: 
	    word = word.lower()
	    if word not in stopwords.words('english') and word not in stopwords_deep_learning: #remove stopwords
	          word_list_cleaned.append(word)  
	    
	  for k in range(len(word_list_cleaned)):
	    keyword_list.append(word_list_cleaned[k])
	  
	keyword_counter = Counter(keyword_list)
	print(keyword_counter)  

	print('{} different keywords before merging'.format(len(keyword_counter)))

	# Merge duplicates: CNNs and CNN
	duplicates = []
	for k in keyword_counter:
	    if k+'s' in keyword_counter:
	        duplicates.append(k)
	for k in duplicates:
	    keyword_counter[k] += keyword_counter[k+'s']
	    del keyword_counter[k+'s']
	print('{} different keywords after merging'.format(len(keyword_counter)))
	with open(conf, 'a', encoding='utf-8') as key_info:
	    key_info.write(str(keyword_counter))

	# Show N most common keywords and their frequencies
	if num_keyword == 0:
	    num_keyword = 50 #FIXME
	keywords_counter_vis = keyword_counter.most_common(num_keyword)
	plt.rcdefaults()
	fig, ax = plt.subplots(figsize=(8, 18))

	key = [k[0] for k in keywords_counter_vis] 
	value = [k[1] for k in keywords_counter_vis] 
	y_pos = np.arange(len(key))
	ax.barh(y_pos, value, align='center', color='green', ecolor='black', log=True)
	ax.set_yticks(y_pos)
	ax.set_yticklabels(key, rotation=0, fontsize=10)
	ax.invert_yaxis() 
	for i, v in enumerate(value):
	    ax.text(v + 3, i + .25, str(v), color='black', fontsize=10)
	ax.set_xlabel('Frequency')
	ax.set_title(conf + ' ' + re.search('\d{2,}', http_list[0]).group() + ' Top {} Keywords'.format(num_keyword))
	plt.show()

	# Show the word cloud forming by keywords
	wordcloud = WordCloud(max_font_size=64, max_words=160, 
	                      width=1280, height=640,
	                      background_color="black").generate(' '.join(keyword_list))
	plt.figure(figsize=(16, 8))
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.show()
	return keyword_counter

'''to find newly appeared words'''
def newWord(tgt_url_list, conf_list, stopwords_deep_learning):
	counter_list = [keyWordVis(tul, cl, stopwords_deep_learning) for tul, cl in zip(tgt_url_list, conf_list)]
	for ci in range(len(counter_list)-1):
		old_c = counter_list[ci]
		new_c =counter_list[ci+1]
		add_c = new_c.keys() - old_c.keys()
		add_m = {ac : new_c[ac] * (new_c[ac] - old_c[ac]) / old_c[ac] for ac in add_c if old_c.get(ac, None) is not None}
		add_m.update({ac : new_c[ac] * (new_c[ac] - 1) for ac in add_c if old_c.get(ac, None) is None})
		add_m = sorted(add_m.items(), key=lambda x : x[1], reverse=True)
		print(add_m[:50]) 

if __name__ == '__main__':
	'''CVPR/ICCV/ECCV'''
	# # tgt_url = ["http://openaccess.thecvf.com/ECCV2018.py"]
	# # tgt_url = ["http://openaccess.thecvf.com/ICCV2019.py"]
	# tgt_url = ["http://openaccess.thecvf.com/CVPR2020.py"]
	# stopwords_list = ['learning', 'network', 'neural', 'networks', 
	# 'deep', 'via', 'using', 'convolutional', 
	# 'single', 'image', 'object', 'based',
	# 'supervised', 'model', 'visual', 'data',
	# 'training', 'dataset', 'toward', 'images', 
	# 'towards', 'with', 'without', 'to']
	
	# '''NIPS'''
	# tgt_url = ["https://papers.nips.cc/book/advances-in-neural-information-processing-systems-32-2019"]
	# stopwords_list = ['via', 'using', 'learning', 'data',
	# 'model', 'network', 'deep', 'neural network', 
	# 'nerual', 'deep neural', 'beyond', 'structure',
	# 'modeling', 'analysis', 'aware', 'neural',
	# 'networks', 'based', 'models', 'algorithm', 
	# 'method', 'training', 'policy', 'prediction',
	# 'inference', 'function', 'processes', 'framework',
	# 'supervised', 'toward', 'methods', 'towards']
	 
	'''ICML'''
	tgt_url = ["https://icml.cc/Conferences/2020/AcceptedPapersInitial"] 
	stopwords_list = ['via', 'using', 'learning', 'data',
	'model', 'network', 'deep', 'neural network', 
	'nerual', 'deep neural', 'beyond', 'structure',
	'modeling', 'analysis', 'aware', 'neural',
	'networks', 'based', 'models', 'algorithm', 
	'method', 'training', 'policy', 'prediction',
	'inference', 'function', 'processes', 'framework',
	'supervised', 'toward', 'methods', 'towards', 
	'algorithms', 'features', 'learning', 'approach',
	'functions', 'machine', 'convolutional', 'feature',
	'understanding', 'effect', 'evaluation', 'label',
	'without', 'process', 'structured', 'system',
	'problems', 'labels']
	
	'''ICLR'''
	# tgt_url = ["https://openreview.net/group?id=ICLR.cc/2020/Conference#accept-poster",
	# "https://openreview.net/group?id=ICLR.cc/2020/Conference#accept-spotlight",
	# "https://openreview.net/group?id=ICLR.cc/2020/Conference#accept-talk"]
	# stopwords_list = ['via', 'using', 'learning', 'data',
	# 'model', 'network', 'deep', 'neural network', 
	# 'nerual', 'deep neural', 'beyond', 'structure',
	# 'modeling', 'analysis', 'aware', 'neural',
	# 'networks', 'based', 'models', 'algorithm', 
	# 'method', 'training', 'policy', 'prediction',
	# 'inference', 'function', 'processes', 'framework',
	# 'supervised', 'toward', 'methods', 'towards', 
	# 'algorithms', 'features', 'learning', 'approach',
	# 'functions', 'machine', 'convolutional', 'feature',
	# 'understanding', 'effect', 'evaluation', 'label',
	# 'without', 'process', 'structured', 'system',
	# 'problems', 'labels']

	coresp_re = [cr for cr in conference_re_map.keys() if re.search(cr, tgt_url[0].upper()) is not None]
	keyWordVis(tgt_url, coresp_re[0], stopwords_list)
	# tgtUrlList = [["http://proceedings.mlr.press/v80/"],
	# 			["http://proceedings.mlr.press/v97/"],
	# 			["https://icml.cc/Conferences/2020/AcceptedPapersInitial"]]
	# corespList = ['icml', 'icml', 'ICML']
	# newWord(tgtUrlList, corespList, stopwords_list)