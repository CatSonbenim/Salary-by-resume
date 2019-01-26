from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


def parcing(url, pages):
    all_urls = []
    for page in range(pages):
        url = '%s%d' % (url, page)
        html = urlopen(url).read()
        soup = BeautifulSoup(html, 'lxml').find_all('h2')
        for i in soup:
            obj = str(i.find('a'))
            link = 'https://www.work.ua' + re.findall("/[a-z]*/\d*/", obj)[0]
            all_urls.append(link)
    return all_urls


def get_info(urls):
    info, salaries = [], []
    for url in urls:
        html = urlopen(url).read()
        soup = BeautifulSoup(html, 'lxml')
        r1 = soup.find_all("div", {"class": "card card-indent wordwrap"})[0].get_text()
        r1 = r1.replace('Контактна інформація', '').replace('Шукач приховав свої особисті дані, але ви зможете '
                                                            'надіслати йому повідомлення або запропонувати вакансію, '
                                                            'якщо відкриєте контакти.', '')\
            .replace('Щоб відкрити контакти, увійдіть як роботодавець або зареєструйтеся.', '')
        r1 = r1.replace('Цей шукач вирішив приховати свої особисті дані та контакти, але йому можна'
                        ' надіслати повідомлення або запропонувати вакансію.', '')\
            .replace('Цей шукач вирішив приховати свої особисті дані та контакти. Ви можете зв\'язатися з ним зі '
                     'сторінки %s' % url, '').replace('Особисті дані приховані', '')
        r1 = r1.replace('Зберегти у відгуки', '').replace('Уже у відгуках', '').replace('\n', ' ')
        r1 = r1.replace('.', ' ').replace(',', ' ').replace('-', ' ').replace('/', ' ')
        while '  ' in r1:
            r1 = r1.replace('  ', ' ')

        try:
            money = soup.find("span", {"class": "normal-weight text-muted-print"}).get_text()
        except AttributeError:
            money = '0'
        r1 = r1.replace(money, '')
        info.append(r1)
        salary = []
        for i in money:
            if i.isdigit():
                salary.append(i)
        numb = 0
        le = len(salary)
        for i in range(le):
            numbers = {'0': 0 * 10**(le - i - 1), '1': 1 * 10**(le - i - 1), '2': 2 * 10**(le - i - 1),
                       '3': 3 * 10**(le - i - 1), '4': 4 * 10**(le - i - 1), '5': 5 * 10**(le - i - 1),
                       '6': 6 * 10**(le - i - 1), '7': 7 * 10**(le - i - 1), '8': 8 * 10**(le - i - 1),
                       '9': 9 * 10**(le - i - 1)}
            numb += numbers[salary[i]]
        salaries.append(numb)
    return info, salaries


def tf_idf(info):

    answer = []
    db = pd.read_csv('skills.csv', ';')
    for i in info:
        ans = 0
        ilist = i.split(' ')
        for j in range(len(ilist)):
            try:
                a = db[db.Skill == ilist[j].lower()]
                b = db[db.Skill == ilist[j].lower()].index
                ans += float(a.at[b[0], 'Point'])
            except (AttributeError, KeyError, IndexError, ValueError):
                pass
        answer.append(ans)
    return answer


url1 = 'https://www.work.ua/resumes-vinnytsya-developer/?salaryfrom=3&period=3&page='
urls1 = parcing(url1, 1)
information1, known_salaries1 = get_info(urls1)
weight_of_info1 = tf_idf(information1)
url2 = 'https://www.work.ua/resumes-vinnytsya-developer/?nosalary=1&period=3&page='
urls2 = parcing(url2, 1)
information, unknown_salaries = get_info(urls2)
weight_of_info = tf_idf(information)

df1 = np.zeros([len(weight_of_info1), 2])
df2 = np.zeros([len(weight_of_info), 2])

for i in range(len(weight_of_info1)):
    df1[i, 0] = weight_of_info1[i]
for i in range(len(weight_of_info)):
    df2[i, 0] = weight_of_info[i]

sfr = LinearRegression()
sfr.fit(df1, known_salaries1)

unknown_salaries = sfr.predict(df2)

data_frame = {'Resume': information, 'Predicted salary': unknown_salaries}
data_frame = pd.DataFrame(data_frame)
print(data_frame)
