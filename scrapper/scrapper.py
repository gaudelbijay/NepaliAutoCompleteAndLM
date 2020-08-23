from bs4 import BeautifulSoup
import requests
import os

'''
func crawl_over_page:

Input: url to main pages, html_post_tag in main pages

output: return list of all url for html_post_tag
'''


def crawl_over_page(url, section='h2', starting_section=0, ending_section=-1):
    response = requests.get(url)
    content = BeautifulSoup(response.content, 'html.parser')
    content = content.findAll(section)
    post_url = [each_content.a['href'] for each_content in content[starting_section:ending_section]]
    return post_url


'''
func scrapper:

Input: url from crawl_over_pages function, html_post_tag in post pages, starting_tag_number, 
ending_tag_number

output: write on .txt file containgng all the post 
'''


def scrapper(url, starting_p=3, ending_p=-10, section='p', my_dir='../data', file_to_save='stories.txt'):
    urls = crawl_over_page(url)
    post = ''
    for u in urls:
        response = requests.get(u)
        content = BeautifulSoup(response.content, "html.parser")
        content = content.findAll(section)

        for each_p in content[starting_p:ending_p]:
            post = post + each_p.text

        if not os.path.exists(my_dir):
            os.makedirs(my_dir)

        with open(my_dir + '/' + file_to_save, encoding='utf-8', mode='w') as f:
            f.write(post)

    print(post)


# run the code
if __name__ == '__main__':
    url = 'http://inepal.org/nepalistories/tag/read-nepali-stories-online/?fbclid=IwAR15RcMQqjA6LdJ5wyGKwiC_VdmYu-4FP1X1wBoeKfw0uOvuYEkgg-xJhgo'
    scrapper(url)
