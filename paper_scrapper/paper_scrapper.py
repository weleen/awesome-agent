'''
Author: Yiming Wu yimingwu0@gmail.com
Date: 2025-02-23 16:02:01
LastEditors: Yiming Wu yimingwu0@gmail.com
LastEditTime: 2025-02-23 16:06:53
FilePath: /MyProjects/Awesome/awesome-agent/paper_scrapper/paper_scrapper.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE#
'''

import requests
import datetime

if __main__ == "__main__":
  date = datetime.date.today()
  is_weekend = date.weekday() > 4
  if is_weekend:
    print("Today is weekend, no need to scrapper")
  else:
    print("Today is weekday, scrapper")
    url = "https://papers.cool/arxiv/cs.AI,cs.CL,cs.CV,cs.GR,cs.RO,cs.LG,stat.ML?date={date}&show=500&sort=1".format(date=date.strftime("%Y-%m-%d"))
    response = requests.get(url)
    print(response.text)