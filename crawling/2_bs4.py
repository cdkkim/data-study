import requests
from bs4 import BeautifulSoup

import pdb

url = 'https://en.wikipedia.org/wiki/Machine_learning'

page = requests.get(url).text
soup = BeautifulSoup(page, 'html.parser')

sidebar = soup.find_all(class_='sidebar')

items_1 = [a.text for a in sidebar[0].find_all('li')]
items_2 = [a.text for a in sidebar[1].find_all('li')]

print(items_1)

pdb.set_trace()
